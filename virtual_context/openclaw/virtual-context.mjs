import { execSync, exec } from "node:child_process";

export default {
  id: "virtual-context",
  register(api) {
    const config = api.getConfig?.() ?? {};
    const bin = config.pythonBin ?? "virtual-context";
    const configFlag = config.configPath ? `-c "${config.configPath}"` : "";

    // before_context_send: sync hook — inject retrieved context
    api.on("before_context_send", (event) => {
      const lastUser = [...event.messages].reverse().find(m => m.role === "user");
      if (!lastUser) return;

      const messageText = typeof lastUser.content === "string"
        ? lastUser.content
        : lastUser.content.map(b => b.text ?? "").join(" ");

      try {
        const stdout = execSync(
          `${bin} ${configFlag} transform --message ${JSON.stringify(messageText)}`,
          { encoding: "utf-8", timeout: 10000 }
        ).trim();

        if (!stdout) return;

        // Inject as system message at position 1 (after existing system)
        const injected = {
          role: "user",
          content: `<virtual-context>\n${stdout}\n</virtual-context>`,
          timestamp: Date.now(),
        };

        const messages = [...event.messages];
        messages.splice(1, 0, injected);
        return { messages };
      } catch (e) {
        // Silently fail — don't block the LLM call
        return;
      }
    }, { priority: 5 });

    // agent_end: async — compact completed conversation
    api.on("agent_end", (event) => {
      if (!event.messages?.length) return;

      const jsonMessages = JSON.stringify(
        event.messages.map(m => ({
          role: m.role,
          content: typeof m.content === "string" ? m.content : JSON.stringify(m.content),
        }))
      );

      // Fire-and-forget: compact in background
      exec(
        `echo '${jsonMessages.replace(/'/g, "'\\''")}' | ${bin} ${configFlag} compact --input -`,
        { timeout: 30000 },
        (err) => {
          if (err) console.error("[virtual-context] compact error:", err.message);
        }
      );
    });
  },
};
