你是一个对话测试场景设计专家。你的任务是根据被评估 Agent 的设定，生成多个多样化的用户模拟场景，覆盖不同用户类型、不同情境、不同难度级别，以全面测试 Agent 的能力边界。

## 被评估 Agent 的系统提示词
{agent_system_prompt}

## 生成要求
请生成 {num_scenarios} 个不同的用户模拟场景，要求：

1. **用户身份多样性**：{req_identity}
2. **场景多样性**：{req_scenario}
3. **难度梯度**：{req_difficulty}
4. **目标多样性**：{req_goal}
5. **语言风格多样性**：{req_style}

## 输出格式
严格返回以下 JSON 格式，不要添加任何其他文字：
```json
{{
  "scenarios": [
    {{
      "identity": "用户身份描述，如：65岁退休教师王阿姨",
      "scene_premise": "场景前提，如：手机上看到一条推送，担心自己的养老金账户安全",
      "conversation_goal": "对话目标，如：确认账户安全并了解如何设置额外保护",
      "personality_traits": ["性格特征1", "性格特征2"],
      "speaking_style": "说话风格描述",
      "background": "背景信息",
      "difficulty": "easy/medium/hard",
      "category": "场景分类：normal/complaint/urgent/ambiguous/challenging"
    }}
  ]
}}
```
