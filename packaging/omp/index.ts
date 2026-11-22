import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

export default function songe(pi: ExtensionAPI) {
	const port = Number(process.env.SONGE_PORT ?? 7613);
	if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error("Invalid SONGE_PORT");
	const baseUrl = `http://127.0.0.1:${port}/v1`;
	pi.registerProvider("songe", {
		api: "openai-completions",
		apiKey: "local",
		baseUrl,
		async fetchDynamicModels() {
			const response = await fetch(`${baseUrl}/models`, { signal: AbortSignal.timeout(5000), redirect: "error" });
			if (!response.ok) throw new Error(`Songe model discovery: HTTP ${response.status}`);
			const model = pi.zod.object({ id: pi.zod.string(), max_model_len: pi.zod.number().int().min(2).optional() });
			const { data } = pi.zod.object({ data: pi.zod.array(model) }).parse(await response.json());
			return data.map(({ id, max_model_len = 8192 }) => ({
				id, name: id, reasoning: false, input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: max_model_len, maxTokens: Math.min(4096, Math.floor(max_model_len / 2)),
			}));
		},
	});
}
