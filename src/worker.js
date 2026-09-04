// 放在代码最上方
const GEMINI_MODELS = {
  // 官方完整名称 (推荐在 Chatbox 里直接用这些)
  "gemini-3-flash-preview": "gemini-3-flash-preview",
  "gemini-2.5-flash": "gemini-2.5-flash",
  "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
  "gemini-2.5-pro": "gemini-2.5-pro",
  "gemini-2.0-flash": "gemini-2.0-flash",
  "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
  
  // 简短别名 (方便手打)
  "gemini-flash": "gemini-3-flash-preview", 
  "gemini-pro": "gemini-2.5-pro",
};

const DEFAULT_OPENAI_MODEL = "gpt-4o"; // 建议改为一个通用的默认模型

const DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20241022"; // 建议使用真实存在的版本

const DEFAULT_UPSTREAM_BASE_URL = "https://apihub.agnes-ai.com/v1";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
  "Access-Control-Allow-Headers": "authorization,content-type,x-api-key,anthropic-api-key,anthropic-version,anthropic-beta,openai-beta",
  "Access-Control-Expose-Headers": "content-type,request-id,x-request-id",
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    const url = new URL(request.url);
    const path = normalizePath(url.pathname);

    try {
      const authError = validateWorkerApiKey(request, env);
      if (authError) return authError;

      if (path === "/" || path === "/health") {
        return jsonResponse(serviceInfo(request, env));
      }

      if (path.startsWith("/api/")) {
        return proxyUpstream(request, env, path);
      }

      if (path === "/mcp" || path === "/v1/mcp" || path === "/anthropic/mcp" || path === "/anthropic/v1/mcp") {
        return jsonResponse(mcpInfo(request));
      }

      if (path === "/codex" || path === "/v1/codex" || path === "/anthropic/codex" || path === "/anthropic/v1/codex") {
        return textResponse(codexSetup(request), "text/plain; charset=utf-8");
      }

      if (path === "/v1/setup" || path === "/anthropic/setup" || path === "/anthropic/v1/setup") {
        return textResponse(agentSetup(request), "text/plain; charset=utf-8");
      }

      if (path === "/v1/messages" || (path === "/v1/models" && looksLikeAnthropicRequest(request)) || path.startsWith("/anthropic/")) {
        return handleAnthropic(request, env, path);
      }

      if (path.startsWith("/v1/")) {
        return handleOpenAI(request, env, path);
      }

      return errorResponse(404, "not_found", `No route for ${path}`);
    } catch (error) {
      return errorResponse(500, "internal_error", error && error.message ? error.message : String(error));
    }
  },
};

async function handleOpenAI(request, env, path) {
  if ((path === "/v1/key" || path === "/v1/auth-key" || path === "/v1/usage") && request.method === "GET") {
    const rawPath = path === "/v1/usage" ? "/api/usage" : "/api/key";
    return proxyUpstream(request, env, rawPath);
  }

  if (path === "/v1/models" && request.method === "GET") {
    if (looksLikeAnthropicRequest(request)) {
      return anthropicModels(request, env);
    }
    return openAIModels(request, env);
  }

  if (path === "/v1/search" && request.method === "POST") {
    const body = await readJson(request);
    return openAIDirectCapability(request, env, body, "/api/search");
  }

  if (path === "/v1/merge" && request.method === "POST") {
    const body = await readJson(request);
    return openAIDirectCapability(request, env, body, "/api/merge");
  }

  if (path === "/v1/chat/completions" && request.method === "POST") {
    const body = await readJson(request);
    return openAIChatCompletions(request, env, body);
  }

  if (path === "/v1/responses" && request.method === "POST") {
    const body = await readJson(request);
    return openAIResponses(request, env, body);
  }

  if (path === "/v1/files" && request.method === "GET") {
    return jsonResponse({ object: "list", data: [], has_more: false });
  }

  if (path === "/v1/files" && request.method === "POST") {
    return openAIFileUpload(request, env);
  }

  if ((path === "/v1/files/extract" || path === "/v1/attachments/extract") && request.method === "POST") {
    const body = await readJson(request);
    const extracted = await callUnlimitedJson(request, env, "/api/attachments/extract", body);
    return jsonResponse(extracted);
  }

  if (path.startsWith("/v1/files/") && request.method === "GET") {
    return errorResponse(404, "not_found", "This Worker is stateless. Bind KV/R2 if you need persisted OpenAI file retrieval.");
  }

  if (path === "/v1/embeddings" || path.startsWith("/v1/audio/") || path.startsWith("/v1/images/")) {
    return errorResponse(501, "unsupported_endpoint", `${path} is not exposed by unlimited.surf and cannot be emulated faithfully.`);
  }

  return errorResponse(404, "not_found", `Unsupported OpenAI-compatible route ${path}`);
}

async function openAIDirectCapability(request, env, body, route) {
  const model = body.model || env.DEFAULT_MODEL || DEFAULT_OPENAI_MODEL;
  const created = nowSeconds();
  const id = `chatcmpl_${randomId()}`;
  const payload = buildUnlimitedPayload({ ...body, web_search: route === "/api/search", merge: route === "/api/merge" }, route);

  if (body.stream !== false) {
    const upstream = await callUnlimitedStream(request, env, route, payload);
    return sseResponse(streamOpenAIChat(upstream, { id, created, model }));
  }

  const result = await collectUnlimitedText(request, env, route, payload);
  return jsonResponse({
    id,
    object: "chat.completion",
    created,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: result.text },
        logprobs: null,
        finish_reason: result.finishReason || "stop",
      },
    ],
    usage: usageFromText(payload.message || payload.query || "", result.text),
    system_fingerprint: `unlimited-surf-worker:${route}`,
  });
}

async function openAIChatCompletions(request, env, body) {
  const model = body.model || env.DEFAULT_MODEL || DEFAULT_OPENAI_MODEL;
  
  // 【核心拦截】如果模型在 GEMINI_MODELS 里，走直连
  if (GEMINI_MODELS[model]) {
    return geminiChatCompletions(body, env, GEMINI_MODELS[model], model);
  }

  // 下面是原来的 Agnes 逻辑
  const created = nowSeconds();
  const id = `chatcmpl_${randomId()}`;
  const route = chooseUnlimitedRoute(body);
  const payload = buildUnlimitedPayload(body, route);

  if (body.stream) {
    const upstream = await callUnlimitedStream(request, env, route, payload);
    return sseResponse(streamOpenAIChat(upstream, { id, created, model }));
  }

  const result = await collectUnlimitedText(request, env, route, payload);
  return jsonResponse({
    id,
    object: "chat.completion",
    created,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: result.text },
        logprobs: null,
        finish_reason: result.finishReason || "stop",
      },
    ],
    usage: usageFromText(payload.message || "", result.text),
    system_fingerprint: "unlimited-surf-worker",
  });
}

async function openAIResponses(request, env, body) {
  const model = body.model || env.DEFAULT_MODEL || DEFAULT_OPENAI_MODEL;
  const created = nowSeconds();
  const id = `resp_${randomId()}`;
  const syntheticChatBody = responsesToChatBody(body, model);
  const route = chooseUnlimitedRoute(syntheticChatBody);
  const payload = buildUnlimitedPayload(syntheticChatBody, route);

  if (body.stream) {
    const upstream = await callUnlimitedStream(request, env, route, payload);
    return sseResponse(streamOpenAIResponses(upstream, { id, created, model }));
  }

  const result = await collectUnlimitedText(request, env, route, payload);
  return jsonResponse({
    id,
    object: "response",
    created_at: created,
    status: "completed",
    error: null,
    incomplete_details: null,
    instructions: body.instructions || null,
    max_output_tokens: body.max_output_tokens || body.max_tokens || null,
    model,
    output: [
      {
        id: `msg_${randomId()}`,
        type: "message",
        status: "completed",
        role: "assistant",
        content: [{ type: "output_text", text: result.text, annotations: [] }],
      },
    ],
    output_text: result.text,
    parallel_tool_calls: true,
    previous_response_id: body.previous_response_id || null,
    reasoning: body.reasoning || null,
    store: body.store || false,
    temperature: body.temperature || null,
    text: body.text || { format: { type: "text" } },
    tool_choice: body.tool_choice || "auto",
    tools: body.tools || [],
    top_p: body.top_p || null,
    truncation: body.truncation || "disabled",
    usage: responseUsageFromText(payload.message || "", result.text),
    user: body.user || null,
  });
}

async function geminiChatCompletions(body, env, googleModel, clientModel) {
  if (!env.GEMINI_API_KEY) {
    return errorResponse(500, "missing_api_key", "GEMINI_API_KEY is required in environment variables.");
  }

  const isStream = body.stream === true;
  const apiMethod = isStream ? "streamGenerateContent" : "generateContent";
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${googleModel}:${apiMethod}?key=${env.GEMINI_API_KEY}${isStream ? "&alt=sse" : ""}`;

  const contents = (body.messages || []).map(message => {
    let role = "user";
    if (message.role === "assistant") role = "model";
    else if (message.role === "system") role = "user";

    const textContent = typeof message.content === "string"
      ? message.content
      : (Array.isArray(message.content)
        ? message.content.map(c => c.type === "text" ? c.text : `[${c.type}]`).join("\n")
        : String(message.content));

    return { role, parts: [{ text: textContent }] };
  });

  const payload = { contents };
  if (body.temperature !== undefined || body.max_tokens !== undefined || body.top_p !== undefined) {
    payload.generationConfig = {};
    if (body.temperature !== undefined) payload.generationConfig.temperature = body.temperature;
    if (body.max_tokens !== undefined) payload.generationConfig.maxOutputTokens = body.max_tokens;
    if (body.top_p !== undefined) payload.generationConfig.topP = body.top_p;
  }

  const systemMsg = (body.messages || []).find(m => m.role === "system");
  if (systemMsg) {
    payload.systemInstruction = {
      role: "system",
      parts: [{ text: typeof systemMsg.content === "string" ? systemMsg.content : JSON.stringify(systemMsg.content) }]
    };
  }

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errText = await response.text();
    console.error("Gemini API Error:", response.status, errText);
    return errorResponse(response.status, "gemini_api_error", `Gemini API failed: ${response.status} - ${errText}`);
  }

  if (isStream) {
    return sseResponse(streamGeminiToOpenAI(response, {
      id: `chatcmpl_${randomId()}`,
      created: nowSeconds(),
      model: clientModel
    }));
  }

  const data = await response.json();
  const candidate = data.candidates?.[0];

  if (!candidate || !candidate.content?.parts?.[0]?.text) {
    const blocked = candidate?.finishReason === "SAFETY";
    return errorResponse(blocked ? 400 : 500, blocked ? "content_blocked" : "gemini_empty_response", blocked ? "Blocked by safety filters." : "Empty response.");
  }

  const text = candidate.content.parts[0].text;
  const finishReason = mapGeminiFinishReason(candidate.finishReason);

  const usage = {
    prompt_tokens: data.usageMetadata?.promptTokenCount || estimateTokens(JSON.stringify(contents)),
    completion_tokens: data.usageMetadata?.candidatesTokenCount || estimateTokens(text),
    total_tokens: data.usageMetadata?.totalTokenCount || (data.usageMetadata?.promptTokenCount + data.usageMetadata?.candidatesTokenCount) || estimateTokens(JSON.stringify(contents) + text)
  };

  return jsonResponse({
    id: `chatcmpl_${randomId()}`,
    object: "chat.completion",
    created: nowSeconds(),
    model: clientModel,
    choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: finishReason }],
    usage,
    system_fingerprint: `direct-gemini:${googleModel}`
  });
}

function streamGeminiToOpenAI(upstream, meta) {
  const decoder = new TextDecoder();
  return new ReadableStream({
    async start(controller) {
      writeSse(controller, { id: meta.id, object: "chat.completion.chunk", created: meta.created, model: meta.model, choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }] });
      try {
        const reader = upstream.body.getReader();
        let buffer = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split(/\r?\n/);
          buffer = lines.pop() || "";
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6).trim();
            if (jsonStr === "[DONE]") continue;
            try {
              const data = JSON.parse(jsonStr);
              const candidate = data.candidates?.[0];
              if (candidate?.content?.parts?.[0]?.text) {
                writeSse(controller, { id: meta.id, object: "chat.completion.chunk", created: meta.created, model: meta.model, choices: [{ index: 0, delta: { content: candidate.content.parts[0].text }, finish_reason: null }] });
              }
              if (candidate?.finishReason) {
                writeSse(controller, { id: meta.id, object: "chat.completion.chunk", created: meta.created, model: meta.model, choices: [{ index: 0, delta: {}, finish_reason: mapGeminiFinishReason(candidate.finishReason) }] });
                writeRawSse(controller, "data: [DONE]\n\n");
              }
            } catch (e) { console.warn("Parse SSE error:", e); }
          }
        }
        writeRawSse(controller, "data: [DONE]\n\n");
      } catch (error) { console.error("Stream error:", error); } finally { controller.close(); }
    }
  });
}

function mapGeminiFinishReason(reason) {
  if (!reason) return "stop";
  switch (reason) {
    case "STOP": 
      return "stop";
    case "MAX_TOKENS": 
      return "length";
    case "SAFETY": 
      return "content_filter";
    case "RECITATION": 
      return "content_filter";
    default: 
      return "stop";
  }
}
