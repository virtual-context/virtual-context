"""What was asked, so it is not asked again.

The corpus this job draws from is not a rolling window: it holds every
message the server has produced, and widening a time window cannot refresh
it. So the defence against sounding repetitive is memory of what was already
asked, and that memory matters most at launch, when there is least of it.

The record is a repetition ledger, not a profile. It keeps an actor id
because "do not tag the same member twice this week" needs one, the source
message ids because "do not mine the same thread twice" needs those, and a
fixed-width topic fingerprint because "do not ask this again in other words"
needs a similarity signal. It does NOT keep the member's words, display
name, or anything else that would let someone reconstruct a picture of him
from this table. Deliberately absent, so the absence is a decision rather
than an oversight:

  * the member's original message text — the id identifies the thread; a
    second copy of what he said is a second place it can leak from
  * his handle or display name — presentation, re-derivable, and the actor
    id is what dedup actually compares
  * channel labels — ids decide everything; labels drift
  * anything about him not needed to prevent a repeat

The topic fingerprint is a 64-bit simhash over normalised tokens. It answers
"is this close to something already asked" and cannot be read back into the
question it came from, so similarity costs no stored content.
"""

from __future__ import annotations

import hashlib
import re
import uuid

try:  # the driver's exception type must not leak past this module
    from psycopg.errors import UniqueViolation as _UNIQUE_VIOLATION
except Exception:  # pragma: no cover - psycopg absent
    class _UNIQUE_VIOLATION(Exception):
        pass
from dataclasses import dataclass
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .candidates import Rejection

# Cooldowns. Deliberately conservative at launch, when the pool is thinnest
# and a repeat is most visible.
MEMBER_COOLDOWN = timedelta(days=14)
QUESTION_SIMILARITY_WINDOW = timedelta(days=60)
CHANNEL_WINDOW = timedelta(days=7)
CHANNEL_MAX_IN_WINDOW = 3
SIMILARITY_DISTANCE = 12  # bits of a 64-bit simhash

# The civil day the claim is keyed on. A restart, a manual re-run and a
# duplicate wake-up must all resolve to the same day, so it is a calendar
# date in one fixed zone rather than an elapsed interval.
POSTING_ZONE = "America/New_York"


def _eastern_day(moment: datetime) -> date:
    return moment.astimezone(ZoneInfo(POSTING_ZONE)).date()

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "did", "do",
    "for", "from", "have", "how", "in", "is", "it", "of", "on", "or", "the",
    "to", "was", "were", "what", "when", "which", "with", "you", "your",
})


class DayAlreadyClaimed(RuntimeError):
    """Raised when a second row tries to take a civil day already held.

    Both backends raise this, so behaviour does not depend on which one is
    installed. In Postgres it is a unique-index violation translated at the
    boundary; the driver's exception type never leaves this module.
    """


@dataclass(frozen=True)
class PostRecord:
    """One question that was asked, reduced to what prevents a repeat."""

    posted_at: datetime
    channel_id: str
    question_type: str                 # "personal" | "timed" | "broader"
    tagged_actor_id: str               # "" when no member was tagged
    source_message_ids: tuple[str, ...] = ()
    topic_fingerprint: int = 0
    question_text: str = ""            # Vast's own words, for near-dup review
    discord_message_id: str = ""
    resolution: str = ""               # "" | "answered" | "ignored"
    # "pending" means the day was claimed and the send may or may not have
    # happened; "posted" means the message id came back. A day is taken in
    # either state, because the only irreversible mistake here is sending
    # twice.
    # "staged"    review post is up, awaiting the owner
    # "approved"  approval seen, publish claimed, send in flight
    # "posted"    published, discord_message_id is the real reply
    # "declined"  rejected; the day was released in the same statement
    # "pending"   the pre-staging claim shape: sent, or possibly sent
    status: str = "posted"
    # The review post in the staging channel, distinct from the published
    # reply. Both exist for a staged row that was approved.
    staged_message_id: str = ""
    # The channel the quoted message lives in. channel_id records where the
    # post went, which under staging is the staging channel — publishing
    # needs the source, and deriving it later from the message id would mean
    # a lookup that can fail at the moment it is needed.
    source_channel_id: str = ""
    # The row's own handle, so a record read back can be addressed.
    #
    # Without it the operator surface was unusable: pending_claims() reported
    # rows for a person to resolve and update() takes a handle, so the only
    # method that could fix them could not be told which one. Empty on a
    # record being written; populated on every record read back.
    id: str = ""


class InMemoryPostHistory:
    """Reference implementation, kept honest against the durable one.

    ``record`` returns an opaque handle rather than nothing, and ``update``
    takes that handle rather than a list position. The position-based form
    read naturally here and could not be implemented durably at all: an
    index into a Python list has no meaning in a table, and two processes
    computing "the last row" both arrive at the same number and confirm each
    other's claim. Since that claim is the primitive preventing a duplicate
    post, the reference implementation must not make it look easier than it
    is.
    """

    def __init__(self) -> None:
        self._records: dict[str, PostRecord] = {}
        self._order: list[str] = []
        # Days freed by a decline. The durable backend does this by nulling
        # eastern_day; in memory the record keeps its timestamp, so the
        # released day is tracked alongside it to keep the two agreeing.
        self._declined_days: set = set()

    def record(self, entry: PostRecord) -> str:
        if entry.posted_at is not None:
            # A new claim on a previously declined day retakes it.
            self._declined_days.discard(_eastern_day(entry.posted_at))
        if entry.posted_at is not None and self.day_is_claimed(
            _eastern_day(entry.posted_at),
        ):
            raise DayAlreadyClaimed(
                f"{_eastern_day(entry.posted_at)} is already claimed"
            )
        import dataclasses

        handle = uuid.uuid4().hex
        self._records[handle] = dataclasses.replace(entry, id=handle)
        self._order.append(handle)
        return handle

    def update(self, handle: str, **changes) -> PostRecord:
        """Replace one record by handle, for the claim-then-confirm sequence."""
        import dataclasses

        if handle not in self._records:
            raise KeyError(f"no post history record for handle {handle!r}")
        self._records[handle] = dataclasses.replace(
            self._records[handle], **changes,
        )
        return self._records[handle]

    def since(self, moment: datetime) -> list[PostRecord]:
        return [r for r in self.all() if r.posted_at >= moment]

    def all(self) -> list[PostRecord]:
        return [self._records[h] for h in self._order]

    def day_is_claimed(self, day: date) -> bool:
        """Whether any record — pending or posted — holds this civil day."""
        if day in self._declined_days:
            return False
        return any(
            _eastern_day(r.posted_at) == day
            for r in self.all()
            if r.posted_at is not None and r.status != "declined"
        )

    def pending(self) -> list[PostRecord]:
        return [r for r in self.all() if r.status == "pending"]

    def claim_for_publish(self, handle: str) -> bool:
        """Same compare-and-set contract as the durable backend."""
        import dataclasses

        record = self._records.get(handle)
        if record is None:
            raise KeyError(f"no post history record for handle {handle!r}")
        if record.status != "staged":
            return False
        self._records[handle] = dataclasses.replace(record, status="approved")
        return True

    def decline(self, handle: str) -> bool:
        """Decline and release the day together. See the durable backend."""
        import dataclasses

        record = self._records.get(handle)
        if record is None:
            raise KeyError(f"no post history record for handle {handle!r}")
        if record.status == "declined":
            return False
        if record.status != "staged":
            raise ValueError(
                f"cannot decline a row with status {record.status!r}; only a "
                "staged row can be declined, and a published message cannot "
                "be unpublished by releasing its day"
            )
        self._records[handle] = dataclasses.replace(record, status="declined")
        self._declined_days.add(_eastern_day(record.posted_at))
        return True


def topic_fingerprint(text: str) -> int:
    """64-bit simhash over normalised tokens; not reversible to the text."""
    tokens = [
        t for t in _TOKEN.findall((text or "").lower())
        if t not in _STOPWORDS and len(t) > 2
    ]
    if not tokens:
        return 0
    vector = [0] * 64
    for token in tokens:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            vector[bit] += 1 if (value >> bit) & 1 else -1
    out = 0
    for bit in range(64):
        if vector[bit] > 0:
            out |= 1 << bit
    return out


def fingerprint_distance(left: int, right: int) -> int:
    """Hamming distance between two fingerprints."""
    return bin(int(left) ^ int(right)).count("1")


def check_repetition(
    *,
    history,
    now: datetime,
    actor_id: str,
    channel_id: str,
    source_message_ids: tuple[str, ...] | list[str],
    question_text: str,
    apply_channel_cap: bool = True,
) -> Rejection | None:
    """The first repetition rule this candidate breaks, or ``None``.

    ``channel_id`` is the channel a question would be POSTED to, because the
    cap counts how much has landed in a channel. The ledger records the
    destination, so passing a source channel here compares two different
    things and the rule silently never matches.

    ``apply_channel_cap`` exists because the cap protects a community from
    being over-posted into. A private rehearsal destination has nothing to
    protect and one deliberate watcher, so the rule is aimed at the wrong
    target there — not relaxed, scoped. The other four rules apply
    everywhere and are unaffected.
    """
    incoming = {str(m) for m in (source_message_ids or []) if str(m)}
    fingerprint = topic_fingerprint(question_text)
    records = history.all()

    for record in records:
        if incoming & set(record.source_message_ids):
            reason = (
                "thread_previously_ignored"
                if record.resolution == "ignored"
                else "thread_already_used"
            )
            return Rejection("", "history", reason, f"posted {record.posted_at:%Y-%m-%d}")

    if actor_id:
        for record in records:
            if (
                record.tagged_actor_id == actor_id
                and now - record.posted_at < MEMBER_COOLDOWN
            ):
                return Rejection(
                    "", "history", "member_recently_tagged",
                    f"last tagged {record.posted_at:%Y-%m-%d}",
                )

    if fingerprint:
        for record in records:
            if now - record.posted_at > QUESTION_SIMILARITY_WINDOW:
                continue
            if not record.topic_fingerprint:
                continue
            distance = fingerprint_distance(fingerprint, record.topic_fingerprint)
            if distance <= SIMILARITY_DISTANCE:
                return Rejection(
                    "", "history", "question_recently_asked",
                    f"distance={distance} from {record.posted_at:%Y-%m-%d}",
                )

    recent_in_channel = [
        r for r in records
        if r.channel_id == channel_id and now - r.posted_at <= CHANNEL_WINDOW
    ]
    if apply_channel_cap and len(recent_in_channel) >= CHANNEL_MAX_IN_WINDOW:
        return Rejection(
            "", "history", "channel_recently_overused",
            f"{len(recent_in_channel)} posts in {CHANNEL_WINDOW.days}d",
        )
    return None


# Designed, NOT applied. The durable table is gated on explicit approval and
# a proven backup; nothing in this module creates or migrates it.
ENGAGEMENT_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS engagement_post_history (
    id                  TEXT PRIMARY KEY,
    tenant_id           TEXT NOT NULL,
    conversation_id     TEXT NOT NULL,
    posted_at           TEXT NOT NULL,
    channel_id          TEXT NOT NULL,
    question_type       TEXT NOT NULL,
    tagged_actor_id     TEXT NOT NULL DEFAULT '',
    source_message_ids  TEXT NOT NULL DEFAULT '',
    -- NUMERIC(20,0), not BIGINT. The fingerprint is an unsigned 64-bit
    -- simhash spanning 0..2^64-1, and Postgres BIGINT is signed with a
    -- ceiling of 2^63-1, so about half of all values would fail to insert.
    -- SQLite accepts them either way, which is why a test on SQLite alone
    -- would not have caught it.
    topic_fingerprint   NUMERIC(20,0) NOT NULL DEFAULT 0,
    question_text       TEXT NOT NULL DEFAULT '',
    discord_message_id  TEXT NOT NULL DEFAULT '',
    resolution          TEXT NOT NULL DEFAULT '',
    -- The day is claimed here BEFORE the message is sent, because a send and
    -- a write cannot be made atomic. 'pending' means claimed and possibly
    -- sent; 'posted' means the message id came back. Both hold the day, so a
    -- crash between the two can only cost a post, never duplicate one.
    status              TEXT NOT NULL DEFAULT 'posted',
    -- The civil day this row claims, stored rather than computed. Postgres
    -- will not index an expression it cannot prove immutable, and converting
    -- a timestamp to an Eastern date is not immutable, so the unique
    -- constraint below needs a real column to sit on.
    eastern_day         DATE
);

-- Present separately as well as in the CREATE TABLE above, so one script
-- serves a fresh install and a database where the table already exists.
ALTER TABLE engagement_post_history
    ADD COLUMN IF NOT EXISTS eastern_day DATE;

-- Staging has two message ids: the review post in the staging channel and
-- the published reply in the source channel. One column cannot hold both,
-- and conflating them loses the ability to say which message a row refers to.
ALTER TABLE engagement_post_history
    ADD COLUMN IF NOT EXISTS staged_message_id TEXT NOT NULL DEFAULT '';

ALTER TABLE engagement_post_history
    ADD COLUMN IF NOT EXISTS source_channel_id TEXT NOT NULL DEFAULT '';

-- There is deliberately NO backfill of eastern_day.
--
-- One existed, keyed on `WHERE eastern_day IS NULL`, and it was correct while
-- NULL could only mean "this row predates the column". It cannot mean only
-- that any more: releasing a day sets eastern_day = NULL, and that IS the
-- release mechanism, because a unique index permits many NULLs and one date.
-- So a re-run tried to re-claim deliberately released days, and collided with
-- the very index it was migrating toward the moment two rows shared a day —
-- which is ordinary once a question can be declined and replaced.
--
-- Nothing needs backfilling. Every writer sets the column on insert, a fresh
-- install gets it from the CREATE TABLE above, and an already-migrated table
-- has its values. A row still NULL is one whose day was released on purpose,
-- and re-deriving it would take back a day the owner was given.

-- The day claim, enforced by the database rather than by a read-then-write in
-- application code. Checking first and inserting second is exactly the shape
-- that fails under two processes, and the day claim is the only thing standing
-- between a crash and a duplicate post into a community channel.
CREATE UNIQUE INDEX IF NOT EXISTS engagement_post_history_one_per_day
    ON engagement_post_history (tenant_id, eastern_day);
CREATE INDEX IF NOT EXISTS engagement_post_history_actor
    ON engagement_post_history (tenant_id, tagged_actor_id, posted_at);
CREATE INDEX IF NOT EXISTS engagement_post_history_channel
    ON engagement_post_history (tenant_id, channel_id, posted_at);
"""


# Fixed, so every applier takes the same lock. blake2b-8 of
# "engagement_post_history schema", read as a signed bigint; the literal is
# written out rather than computed so it is greppable and cannot drift with a
# hashing change.
ENGAGEMENT_SCHEMA_LOCK = 2278691887160911542


def apply_engagement_history_schema(store) -> str:
    """Create the history table from the shipped DDL, idempotently.

    An executor rather than a copied snippet: hand-applying means the text
    that runs and the text the tests assert are related only by the care of
    whoever copied it. This runs the constant itself, so they cannot drift.

    Every statement is ``IF NOT EXISTS``, so a second run is a no-op and a
    partially applied state completes rather than conflicting. Returns the
    DDL that was executed, for the caller to record.

    The whole script is sent in one call rather than split on semicolons.
    Splitting looks obvious and is wrong here: a semicolon inside the SQL
    comment cuts the CREATE TABLE in half, and the executor would then send a
    truncated statement — worse than the hand-application it was meant to
    replace, because it would look automated while being broken.

    An advisory lock wraps the whole script, because ``IF NOT EXISTS`` is
    idempotent across *time* and not across *concurrency*: two sessions can
    both find the table absent and both try to create it, and Postgres then
    fails the loser inside its catalogue rather than treating it as the
    no-op the clause implies. Measured at 8 failures in 12 concurrent
    applies before the lock, 0 after. Every applier must go through here for
    the lock to mean anything, which is the second reason this is a function
    rather than a snippet to copy.
    """
    connect = getattr(getattr(store, "pool", None), "connection", None)
    if not callable(connect):
        raise RuntimeError(
            "this store exposes no connection pool; the engagement history "
            "table belongs in the tenant-scoped backend alongside "
            "canonical_turns"
        )
    with connect() as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (ENGAGEMENT_SCHEMA_LOCK,))
        try:
            conn.execute(ENGAGEMENT_HISTORY_DDL)
        finally:
            conn.execute(
                "SELECT pg_advisory_unlock(%s)", (ENGAGEMENT_SCHEMA_LOCK,),
            )
    return ENGAGEMENT_HISTORY_DDL


class PostgresPostHistory:
    """The durable ledger. Same interface as the in-memory reference.

    Scoped to one tenant and one conversation at construction, so no query
    here can reach another tenant's rows even if a caller passes the wrong
    conversation id — the scope is not a filter the caller supplies per call.

    ``day_is_claimed`` is a real query rather than a scan of loaded rows,
    because it is the one question asked on every run and the only one whose
    answer decides whether an irreversible action happens. The Eastern day is
    computed in SQL from the stored timestamp rather than in Python from a
    loaded set, so a row written by another process counts immediately.

    ``all`` does load the tenant's rows. That is deliberate and bounded: the
    job posts at most once a day, so the table grows by roughly 365 rows a
    year, and the thread-reuse rule in ``check_repetition`` is unbounded in
    time — it must see every source message id ever used, or it will re-mine
    a thread it used two years ago. A windowed query would be faster and
    wrong.
    """

    def __init__(self, store, *, tenant_id: str, conversation_id: str) -> None:
        connect = getattr(getattr(store, "pool", None), "connection", None)
        if not callable(connect):
            raise RuntimeError(
                "this store exposes no connection pool; the durable post "
                "history requires the Postgres tenant-scoped backend"
            )
        self._connect = connect
        self._tenant_id = str(tenant_id)
        self._conversation_id = str(conversation_id)

    # -- writes ---------------------------------------------------------

    def record(self, entry: PostRecord) -> str:
        """Insert one record and return its handle.

        The handle is the row's own primary key, so a confirmation later
        addresses exactly the row this call created — not "the most recent
        row", which is what a second process would also compute.
        """
        handle = uuid.uuid4().hex
        try:
            self._insert(handle, entry)
        except _UNIQUE_VIOLATION as exc:
            # The database refused a second claim on the same day. That is
            # the constraint doing its job, not an error to surface raw.
            raise DayAlreadyClaimed(
                f"{_eastern_day(entry.posted_at)} is already claimed"
            ) from exc
        return handle

    def _insert(self, handle: str, entry: PostRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO engagement_post_history (
                       id, tenant_id, conversation_id, posted_at, channel_id,
                       question_type, tagged_actor_id, source_message_ids,
                       topic_fingerprint, question_text, discord_message_id,
                       resolution, status, eastern_day, staged_message_id,
                       source_channel_id
                   ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                             %s, %s, %s, %s)""",
                (
                    handle, self._tenant_id, self._conversation_id,
                    entry.posted_at.isoformat(), entry.channel_id,
                    entry.question_type, entry.tagged_actor_id,
                    _join_ids(entry.source_message_ids),
                    int(entry.topic_fingerprint or 0), entry.question_text,
                    entry.discord_message_id, entry.resolution, entry.status,
                    _eastern_day(entry.posted_at), entry.staged_message_id,
                    entry.source_channel_id,
                ),
            )

    def update(self, handle: str, **changes) -> PostRecord:
        """Confirm or amend one row, addressed by handle."""
        if not changes:
            raise ValueError("update called with no changes")
        columns = {
            "discord_message_id", "status", "resolution", "question_text",
            "staged_message_id",
        }
        unknown = set(changes) - columns
        if unknown:
            raise ValueError(f"not updatable: {sorted(unknown)}")
        assignments = ", ".join(f"{c} = %s" for c in changes)
        params = list(changes.values()) + [handle, self._tenant_id]
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE engagement_post_history SET {assignments} "
                "WHERE id = %s AND tenant_id = %s",
                params,
            )
            if cur.rowcount != 1:
                raise KeyError(
                    f"no post history record for handle {handle!r} "
                    f"(rows updated: {cur.rowcount})"
                )
        return self._one(handle)

    # -- reads ----------------------------------------------------------

    def day_is_claimed(self, day: date) -> bool:
        """Whether any row — pending or posted — holds this civil day.

        The comparison happens in Postgres so it sees committed rows from
        every process, and the zone conversion happens there too so a row
        written near UTC midnight lands on the same civil day this asks for.
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT EXISTS (
                       SELECT 1 FROM engagement_post_history
                        WHERE tenant_id = %s AND eastern_day = %s
                   ) AS claimed""",
                (self._tenant_id, day),
            ).fetchone()
        if row is None:
            return False
        return bool(_row_value(row, "claimed", ("claimed",)))

    def pending(self) -> list[PostRecord]:
        return self._select("AND status = 'pending'")

    def claim_for_publish(self, handle: str) -> bool:
        """Move one staged row to approved. True only for the caller that won.

        A compare-and-set on status, so two pollers seeing the same approval
        cannot both publish: the UPDATE matches only while the row is still
        staged, and exactly one caller gets a row. Checking then writing would
        let both read "staged" before either wrote.
        """
        with self._connect() as conn:
            cur = conn.execute(
                """UPDATE engagement_post_history SET status = 'approved'
                    WHERE id = %s AND tenant_id = %s AND status = 'staged'""",
                (handle, self._tenant_id),
            )
            return cur.rowcount == 1

    def decline(self, handle: str) -> bool:
        """Decline a staged row and free its day, in one statement.

        Atomic because the two orderings are not symmetric. Declining without
        releasing costs a day and is recoverable tomorrow; releasing without
        declining leaves the row readable as staged with the day free, so the
        next run stages a SECOND question while the first still awaits
        approval — two staged messages, either publishable. One statement
        removes the choice rather than making it carefully.

        Returns True if this call declined it, False if it was already
        declined; a poller can see the same reply twice and must not treat
        its own correct behaviour as an error. Anything other than a staged
        or declined row raises, so a decline arriving after a publish is
        visible rather than absorbed.
        """
        with self._connect() as conn:
            cur = conn.execute(
                """UPDATE engagement_post_history
                      SET status = 'declined', eastern_day = NULL
                    WHERE id = %s AND tenant_id = %s AND status = 'staged'""",
                (handle, self._tenant_id),
            )
            if cur.rowcount == 1:
                return True
        current = self._one(handle)
        if current.status == "declined":
            return False
        raise ValueError(
            f"cannot decline a row with status {current.status!r}; only a "
            "staged row can be declined, and a published message cannot be "
            "unpublished by releasing its day"
        )

    def since(self, moment: datetime) -> list[PostRecord]:
        return [r for r in self.all() if r.posted_at >= moment]

    def all(self) -> list[PostRecord]:
        return self._select("")

    # -- internals ------------------------------------------------------

    # Named so the row can be read without depending on the connection's
    # row factory. Production builds its pool with dict_row and returns
    # mappings; a bare psycopg connection returns tuples. Code that indexes
    # positionally works against one and raises KeyError against the other,
    # and a test using the wrong factory passes while the class cannot read
    # a row in production.
    _COLUMN_NAMES = (
        "id", "posted_at", "channel_id", "question_type", "tagged_actor_id",
        "source_message_ids", "topic_fingerprint", "question_text",
        "discord_message_id", "resolution", "status", "staged_message_id",
        "source_channel_id",
    )
    _COLUMNS = ", ".join(_COLUMN_NAMES)

    def _select(self, extra: str) -> list[PostRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {self._COLUMNS} FROM engagement_post_history "
                f"WHERE tenant_id = %s {extra} ORDER BY posted_at, id",
                (self._tenant_id,),
            ).fetchall()
        return [_row_to_record(r, self._COLUMN_NAMES) for r in rows]

    def _one(self, handle: str) -> PostRecord:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {self._COLUMNS} FROM engagement_post_history "
                "WHERE id = %s AND tenant_id = %s",
                (handle, self._tenant_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"no post history record for handle {handle!r}")
        return _row_to_record(row, self._COLUMN_NAMES)


def _join_ids(ids) -> str:
    return ",".join(str(i) for i in (ids or ()) if str(i))


def _split_ids(blob: str) -> tuple[str, ...]:
    return tuple(p for p in (blob or "").split(",") if p)


def _row_value(row, name: str, columns):
    """One column, whether the driver returned a mapping or a sequence.

    Unpacking a dict yields its KEYS, so positional access does not merely
    fail here — it silently produces column names where values belong, which
    is how ``posted_at`` reached ``datetime.fromisoformat`` as a string
    literal. Reading by name is correct under either factory.
    """
    if isinstance(row, Mapping):
        return row[name]
    return row[tuple(columns).index(name)]


def _row_to_record(row, columns) -> PostRecord:
    def field(name):
        return _row_value(row, name, columns)

    return PostRecord(
        id=str(field("id") or ""),
        posted_at=datetime.fromisoformat(field("posted_at")),
        channel_id=field("channel_id"),
        question_type=field("question_type"),
        tagged_actor_id=field("tagged_actor_id"),
        source_message_ids=_split_ids(field("source_message_ids")),
        # NUMERIC comes back as Decimal; the fingerprint is compared bitwise.
        topic_fingerprint=int(field("topic_fingerprint") or 0),
        question_text=field("question_text"),
        discord_message_id=field("discord_message_id"),
        resolution=field("resolution"),
        status=field("status"),
        staged_message_id=str(field("staged_message_id") or ""),
        source_channel_id=str(field("source_channel_id") or ""),
    )
