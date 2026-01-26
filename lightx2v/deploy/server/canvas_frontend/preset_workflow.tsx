// --- Presets Data ---

import { WorkflowState, NodeStatus } from './types';
import { getAccessToken } from './src/utils/apiClient';

export const PRESET_WORKFLOWS: WorkflowState[] = [
  // 宫崎骏风格治愈动画视频工作流
  {
    id: 'preset-miyazaki-healing-video',
    name: '宫崎骏风格治愈动画视频',
    updatedAt: Date.now(),
    nodes: [
      {
        id: 'node-scene-text',
        toolId: 'text-input',
        x: 100,
        y: 200,
        data: {
          value: '宫崎骏风格的治愈场景：宁静的乡村小屋，远处是连绵的绿色山丘，天空中有几朵白云，温暖的阳光洒在大地上，远处有风车在缓缓转动。整体色调柔和，充满自然气息。'
        },
        status: NodeStatus.IDLE,
        error: null
      },
      {
        id: 'node-scene-image',
        toolId: 'text-to-image',
        x: 500,
        y: 200,
        data: {
          value: '',
          model: 'Qwen-Image-2512',
          aspectRatio: '16:9'
        },
        status: NodeStatus.IDLE,
        error: null
      },
      {
        id: 'node-motion-text',
        toolId: 'text-input',
        x: 500,
        y: 400,
        data: {
          value: '镜头缓慢向前推进，云朵在天空中缓缓飘动，风车叶片缓慢旋转，阳光透过云层洒下光影变化，整体氛围宁静治愈，多个镜头自然过渡'
        },
        status: NodeStatus.IDLE,
        error: null
      },
      {
        id: 'node-video',
        toolId: 'video-gen-image',
        x: 900,
        y: 300,
        data: {
          value: '',
          model: 'Wan2.2_I2V_A14B_distilled'
        },
        status: NodeStatus.IDLE,
        error: null
      }
    ],
    connections: [
      {
        id: 'conn-scene-text-to-image',
        sourceNodeId: 'node-scene-text',
        sourcePortId: 'out-text',
        targetNodeId: 'node-scene-image',
        targetPortId: 'in-text'
      },
      {
        id: 'conn-image-to-video',
        sourceNodeId: 'node-scene-image',
        sourcePortId: 'out-image',
        targetNodeId: 'node-video',
        targetPortId: 'in-image'
      },
      {
        id: 'conn-motion-to-video',
        sourceNodeId: 'node-motion-text',
        sourcePortId: 'out-text',
        targetNodeId: 'node-video',
        targetPortId: 'in-text'
      }
    ],
    isDirty: false,
    isRunning: false,
    globalInputs: {},
    env: {
      lightx2v_url: (process.env.LIGHTX2V_URL || 'https://x2v.light-ai.top').trim(),
      lightx2v_token: (process.env.LIGHTX2V_TOKEN || '').trim()
    },
    history: [],
    showIntermediateResults: true
  },
  {
    id: 'preset-knowledge-ip',
    name: '知识IP口播工作流',
    updatedAt: Date.now(),
    isDirty: false,
    isRunning: false,
    env: {
      lightx2v_url: "",
      lightx2v_token: ""
    },
    globalInputs: {},
    history: [],
    showIntermediateResults: true,
    connections: [
      { id: 'c1', sourceNodeId: 'ip-node-url', sourcePortId: 'out-text', targetNodeId: 'ip-node-ai', targetPortId: 'in-text' },
      { id: 'c2', sourceNodeId: 'ip-node-image-ref', sourcePortId: 'out-image', targetNodeId: 'ip-node-ai', targetPortId: 'in-image' },
      { id: 'c3', sourceNodeId: 'ip-node-ai', sourcePortId: 'tts_text', targetNodeId: 'ip-node-tts', targetPortId: 'in-text' },
      { id: 'c4', sourceNodeId: 'ip-node-ai', sourcePortId: 'voice_style', targetNodeId: 'ip-node-tts', targetPortId: 'in-context-tone' },
      { id: 'c5', sourceNodeId: 'ip-node-image-ref', sourcePortId: 'out-image', targetNodeId: 'ip-node-avatar', targetPortId: 'in-image' },
      { id: 'c6', sourceNodeId: 'ip-node-tts', sourcePortId: 'out-audio', targetNodeId: 'ip-node-avatar', targetPortId: 'in-audio' },
      { id: 'c7', sourceNodeId: 'ip-node-ai', sourcePortId: 's2v_prompt', targetNodeId: 'ip-node-avatar', targetPortId: 'in-text' }
    ],
    nodes: [
      { id: 'ip-node-url', toolId: 'text-input', x: 50, y: 200, status: NodeStatus.IDLE, data: { value: "https://github.com/ModelTC/LightX2V/blob/main/README_zh.md" } },
      { id: 'ip-node-image-ref', toolId: 'image-input', x: 50, y: 400, status: NodeStatus.IDLE, data: { value: ['/assets/programmer.png'] } },
      { id: 'ip-node-ai', toolId: 'text-generation', x: 450, y: 300, status: NodeStatus.IDLE, data: { 
          model: 'doubao-seed-1-6-vision-250815',
          mode: 'custom',
          useSearch: true,
          customInstruction: `你是一位专业的知识IP口播视频创意总监。你的任务是根据输入的网页链接内容和数字人参考图片，生成完整的知识IP口播视频组件。

重要原则：
- 生成结果中的所有字段语言必须跟随用户输入的语言。如果用户使用中文输入，所有输出字段（tts_text、voice_style、s2v_prompt）都必须使用中文；如果用户使用英文输入，则所有输出字段都使用英文。
- 你需要访问用户提供的网页链接，提取网页的核心内容，理解其主题和关键信息。使用联网搜索功能获取网页内容。
- 用户会提供参考图片，直接使用该图片作为数字人形象，不需要生成或修改图片。

对于 tts_text（口播文案）：
- 根据网页链接的内容，提取核心知识点和关键信息
- 将网页内容转化为自然、流畅的口播文案
- 文案应该结构清晰，逻辑性强，易于理解
- 使用口语化的表达方式，适合口播
- 长度应适合数字人视频使用（正常语速下约30-60秒）
- 确保文案准确传达网页的核心内容，同时保持吸引力和可理解性

对于 voice_style（语气指令）：
- 根据知识IP的定位和内容特点，提供相应的配音指导
- 知识IP通常需要专业、清晰、有说服力的语调
- 描述配音应该体现的风格、情感、语气等特点
- 确保语调指令与口播文案内容相匹配
- 可以包含节奏、重音、停顿等具体指导

对于 s2v_prompt（数字人视频动作提示）：
- 描述自然、真实的说话手势和动作
- 包含头部动作（点头、倾斜、轻微转动）
- 描述与语音内容和情感语调匹配的面部表情
- 知识IP可以有一些专业的手势，如指向、展示等
- 指定眼神接触和视线方向
- 确保动作与语音节奏同步
- 足够详细，以指导数字人视频生成自然、逼真的效果`,
          customOutputs: [
            { id: 'tts_text', label: '口播文案', description: '根据网页内容生成的口播文案。' },
            { id: 'voice_style', label: '语气指令', description: '根据知识IP定位生成的配音指导。' },
            { id: 's2v_prompt', label: 'S2V提示词', description: '数字人视频动作和运动的描述。' }
          ]
      } },
      { id: 'ip-node-tts', toolId: 'tts', x: 900, y: 300, status: NodeStatus.IDLE, data: { model: 'lightx2v', voiceType: 'zh_female_vv_uranus_bigtts', resourceId: 'seed-tts-2.0' } },
      { id: 'ip-node-avatar', toolId: 'avatar-gen', x: 1500, y: 200, status: NodeStatus.IDLE, data: { model: 'SekoTalk' } }
    ]
  },
    {
      id: 'preset-product-wearing',
      name: '虚拟人结合电商产品工作流',
      updatedAt: Date.now(),
      isDirty: false,
      isRunning: false,
      env: {
        lightx2v_url: "",
        lightx2v_token: ""
      },
      globalInputs: {},
      history: [],
      showIntermediateResults: true,
      connections: [
        // Input to AI Chat
        { id: 'prod-c1', sourceNodeId: 'prod-node-person', sourcePortId: 'out-image', targetNodeId: 'prod-node-planner', targetPortId: 'in-image' },
        { id: 'prod-c2', sourceNodeId: 'prod-node-product', sourcePortId: 'out-image', targetNodeId: 'prod-node-planner', targetPortId: 'in-image' },
        { id: 'prod-c3', sourceNodeId: 'prod-node-text', sourcePortId: 'out-text', targetNodeId: 'prod-node-planner', targetPortId: 'in-text' },
        // AI Chat to first image-to-image (front view) - both person and product images
        { id: 'prod-c4', sourceNodeId: 'prod-node-planner', sourcePortId: 'front_prompt', targetNodeId: 'prod-node-i2i-front', targetPortId: 'in-text' },
        { id: 'prod-c5', sourceNodeId: 'prod-node-person', sourcePortId: 'out-image', targetNodeId: 'prod-node-i2i-front', targetPortId: 'in-image' },
        { id: 'prod-c5b', sourceNodeId: 'prod-node-product', sourcePortId: 'out-image', targetNodeId: 'prod-node-i2i-front', targetPortId: 'in-image' },
        // All subsequent i2i nodes use the front image as base
        { id: 'prod-c6', sourceNodeId: 'prod-node-i2i-front', sourcePortId: 'out-image', targetNodeId: 'prod-node-i2i-right45', targetPortId: 'in-image' },
        { id: 'prod-c7', sourceNodeId: 'prod-node-planner', sourcePortId: 'right45_prompt', targetNodeId: 'prod-node-i2i-right45', targetPortId: 'in-text' },
        { id: 'prod-c8', sourceNodeId: 'prod-node-i2i-front', sourcePortId: 'out-image', targetNodeId: 'prod-node-i2i-side90', targetPortId: 'in-image' },
        { id: 'prod-c9', sourceNodeId: 'prod-node-planner', sourcePortId: 'side90_prompt', targetNodeId: 'prod-node-i2i-side90', targetPortId: 'in-text' },
        { id: 'prod-c10', sourceNodeId: 'prod-node-i2i-front', sourcePortId: 'out-image', targetNodeId: 'prod-node-i2i-left45', targetPortId: 'in-image' },
        { id: 'prod-c11', sourceNodeId: 'prod-node-planner', sourcePortId: 'left45_prompt', targetNodeId: 'prod-node-i2i-left45', targetPortId: 'in-text' },
        { id: 'prod-c12', sourceNodeId: 'prod-node-i2i-front', sourcePortId: 'out-image', targetNodeId: 'prod-node-i2i-low', targetPortId: 'in-image' },
        { id: 'prod-c13', sourceNodeId: 'prod-node-planner', sourcePortId: 'low_prompt', targetNodeId: 'prod-node-i2i-low', targetPortId: 'in-text' },
        { id: 'prod-c14', sourceNodeId: 'prod-node-i2i-front', sourcePortId: 'out-image', targetNodeId: 'prod-node-i2i-high', targetPortId: 'in-image' },
        { id: 'prod-c15', sourceNodeId: 'prod-node-planner', sourcePortId: 'high_prompt', targetNodeId: 'prod-node-i2i-high', targetPortId: 'in-text' },
        // All images to video generation
        { id: 'prod-c16', sourceNodeId: 'prod-node-i2i-front', sourcePortId: 'out-image', targetNodeId: 'prod-node-video-front', targetPortId: 'in-image' },
        { id: 'prod-c17', sourceNodeId: 'prod-node-planner', sourcePortId: 'front_motion', targetNodeId: 'prod-node-video-front', targetPortId: 'in-text' },
        { id: 'prod-c18', sourceNodeId: 'prod-node-i2i-right45', sourcePortId: 'out-image', targetNodeId: 'prod-node-video-right45', targetPortId: 'in-image' },
        { id: 'prod-c19', sourceNodeId: 'prod-node-planner', sourcePortId: 'right45_motion', targetNodeId: 'prod-node-video-right45', targetPortId: 'in-text' },
        { id: 'prod-c20', sourceNodeId: 'prod-node-i2i-side90', sourcePortId: 'out-image', targetNodeId: 'prod-node-video-side90', targetPortId: 'in-image' },
        { id: 'prod-c21', sourceNodeId: 'prod-node-planner', sourcePortId: 'side90_motion', targetNodeId: 'prod-node-video-side90', targetPortId: 'in-text' },
        { id: 'prod-c22', sourceNodeId: 'prod-node-i2i-left45', sourcePortId: 'out-image', targetNodeId: 'prod-node-video-left45', targetPortId: 'in-image' },
        { id: 'prod-c23', sourceNodeId: 'prod-node-planner', sourcePortId: 'left45_motion', targetNodeId: 'prod-node-video-left45', targetPortId: 'in-text' },
        { id: 'prod-c24', sourceNodeId: 'prod-node-i2i-low', sourcePortId: 'out-image', targetNodeId: 'prod-node-video-low', targetPortId: 'in-image' },
        { id: 'prod-c25', sourceNodeId: 'prod-node-planner', sourcePortId: 'low_motion', targetNodeId: 'prod-node-video-low', targetPortId: 'in-text' },
        { id: 'prod-c26', sourceNodeId: 'prod-node-i2i-high', sourcePortId: 'out-image', targetNodeId: 'prod-node-video-high', targetPortId: 'in-image' },
        { id: 'prod-c27', sourceNodeId: 'prod-node-planner', sourcePortId: 'high_motion', targetNodeId: 'prod-node-video-high', targetPortId: 'in-text' }
      ],
      nodes: [
        // Input nodes
        { id: 'prod-node-person', toolId: 'image-input', x: 50, y: 200, status: NodeStatus.IDLE, data: { value: ['/assets/model_girl.png'] } },
        { id: 'prod-node-product', toolId: 'image-input', x: 50, y: 400, status: NodeStatus.IDLE, data: { value: ['/assets/product_glass.png'] } },
        { id: 'prod-node-text', toolId: 'text-input', x: 50, y: 600, status: NodeStatus.IDLE, data: { value: "" } },
        // AI Chat Planner (Doubao Vision)
        { id: 'prod-node-planner', toolId: 'text-generation', x: 450, y: 400, status: NodeStatus.IDLE, data: { 
            model: 'doubao-seed-1-6-vision-250815',
            mode: 'custom',
            customInstruction: `你是一位专业的虚拟人产品展示创意总监。你的任务是根据输入的人物图片、产品图片和可选的文字描述，生成详细的图生图修改提示词和视频运镜提示词。

  重要原则：
  - 生成结果中的所有字段语言必须跟随用户输入的语言。如果用户使用中文输入，所有输出字段（front_prompt、front_motion、right45_prompt等）都必须使用中文；如果用户使用英文输入，则所有输出字段都使用英文。
  - 保持人物特征的一致性（面部特征、身材、服装等）
  - 产品必须准确、自然地佩戴在人物身上
  - 每个角度的图片都要保持产品佩戴位置和方式的准确性
  - 确保所有角度的图片风格统一

  你需要生成6个角度的图生图修改提示词和对应的视频运镜提示词：

  1. 正面图（front_prompt）：让输入的人物图片中的人物佩戴输入的产品图片中的产品，保持人物和产品的一致性
  2. 右边45度侧面（right45_prompt）：将正面图中的人物向右转45度，微微低头，保持人物和产品的一致性
  3. 90度完全侧面（side90_prompt）：将正面图中的人物完全侧身，保持人物和产品的一致性
  4. 左边45度侧面（left45_prompt）：将正面图中的人物向左转45度，保持人物和产品的一致性
  5. 仰拍角度（low_prompt）：将正面图中的人物改为仰拍角度，从下往上拍摄，保持人物和产品的一致性
  6. 俯拍角度（high_prompt）：将正面图中的人物改为俯拍角度，从上往下拍摄，保持人物和产品的一致性

  每个图生图修改提示词应该：
  - 强调"改"和"一致性"，不需要描述图片本身的内容
  - 明确说明如何修改：例如"让图一的人物佩戴图二的眼镜，保持人物和物品一致性"
  - 描述需要改变的姿势、角度、视角
  - 强调保持人物特征和产品的一致性
  - 不要描述光线、背景、氛围等图片细节

  每个视频运镜提示词应该：
  - 描述简单的相机运动，如"镜头从下移动至上，聚焦模特"
  - 与对应角度的图片内容匹配
  - 简洁明了，适合图生视频使用

  输出格式：JSON，包含以下字段：
  - front_prompt, front_motion
  - right45_prompt, right45_motion
  - side90_prompt, side90_motion
  - left45_prompt, left45_motion
  - low_prompt, low_motion
  - high_prompt, high_motion`,
            customOutputs: [
              { id: 'front_prompt', label: '正面图提示词', description: '人物正面佩戴产品的详细图生图提示词' },
              { id: 'front_motion', label: '正面运镜提示词', description: '正面图的视频运镜提示词' },
              { id: 'right45_prompt', label: '右边45度提示词', description: '右边45度侧面，人物微微低头的图生图提示词' },
              { id: 'right45_motion', label: '右边45度运镜提示词', description: '右边45度图的视频运镜提示词' },
              { id: 'side90_prompt', label: '90度侧面提示词', description: '完全侧面的图生图提示词' },
              { id: 'side90_motion', label: '90度侧面运镜提示词', description: '90度侧面图的视频运镜提示词' },
              { id: 'left45_prompt', label: '左边45度提示词', description: '左边45度侧面的图生图提示词' },
              { id: 'left45_motion', label: '左边45度运镜提示词', description: '左边45度图的视频运镜提示词' },
              { id: 'low_prompt', label: '仰拍提示词', description: '仰拍角度的图生图提示词' },
              { id: 'low_motion', label: '仰拍运镜提示词', description: '仰拍图的视频运镜提示词' },
              { id: 'high_prompt', label: '俯拍提示词', description: '俯拍角度的图生图提示词' },
              { id: 'high_motion', label: '俯拍运镜提示词', description: '俯拍图的视频运镜提示词' }
            ]
        } },
        // Image-to-image nodes (6 angles)
        { id: 'prod-node-i2i-front', toolId: 'image-to-image', x: 900, y: 50, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'prod-node-i2i-right45', toolId: 'image-to-image', x: 900, y: 150, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'prod-node-i2i-side90', toolId: 'image-to-image', x: 900, y: 250, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'prod-node-i2i-left45', toolId: 'image-to-image', x: 900, y: 350, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'prod-node-i2i-low', toolId: 'image-to-image', x: 900, y: 450, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'prod-node-i2i-high', toolId: 'image-to-image', x: 900, y: 550, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        // Video generation nodes (6 videos)
        { id: 'prod-node-video-front', toolId: 'video-gen-image', x: 1500, y: 50, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
        { id: 'prod-node-video-right45', toolId: 'video-gen-image', x: 1500, y: 150, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
        { id: 'prod-node-video-side90', toolId: 'video-gen-image', x: 1500, y: 250, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
        { id: 'prod-node-video-left45', toolId: 'video-gen-image', x: 1500, y: 350, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
        { id: 'prod-node-video-low', toolId: 'video-gen-image', x: 1500, y: 450, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
        { id: 'prod-node-video-high', toolId: 'video-gen-image', x: 1500, y: 550, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } }
      ]
    },
    {
      id: 'preset-avatar-i2i',
      name: '无限数字人口播视频工作流',
      updatedAt: Date.now(),
      isDirty: false,
      isRunning: false,
      env: {
        lightx2v_url: "",
        lightx2v_token: ""
      },
      globalInputs: {},
      history: [],
      showIntermediateResults: true,
      connections: [
        { id: 'c1', sourceNodeId: 'node-img-in', sourcePortId: 'out-image', targetNodeId: 'node-i2i-gen', targetPortId: 'in-image' },
        { id: 'c2', sourceNodeId: 'node-text-in', sourcePortId: 'out-text', targetNodeId: 'node-logic', targetPortId: 'in-text' },
        { id: 'c3', sourceNodeId: 'node-logic', sourcePortId: 'i2i_prompt', targetNodeId: 'node-i2i-gen', targetPortId: 'in-text' },
        { id: 'c4', sourceNodeId: 'node-logic', sourcePortId: 'tts_text', targetNodeId: 'node-voice', targetPortId: 'in-text' },
        { id: 'c5', sourceNodeId: 'node-logic', sourcePortId: 'voice_style', targetNodeId: 'node-voice', targetPortId: 'in-context-tone' },
        { id: 'c6', sourceNodeId: 'node-i2i-gen', sourcePortId: 'out-image', targetNodeId: 'node-final-avatar', targetPortId: 'in-image' },
        { id: 'c7', sourceNodeId: 'node-voice', sourcePortId: 'out-audio', targetNodeId: 'node-final-avatar', targetPortId: 'in-audio' }
      ],
      nodes: [
        { id: 'node-img-in', toolId: 'image-input', x: 50, y: 50, status: NodeStatus.IDLE, data: { value: ['/assets/girl.jpg'] } },
        { id: 'node-text-in', toolId: 'text-input', x: 50, y: 350, status: NodeStatus.IDLE, data: { value: "女孩改为穿着性感纯欲的睡衣坐在床上，用性感迷人的声音说着勾人的话" } },
        { id: 'node-logic', toolId: 'text-generation', x: 450, y: 350, status: NodeStatus.IDLE, data: { 
            model: 'doubao-seed-1-6-vision-250815',
            mode: 'custom',
            customInstruction: `你是一位专业的数字人视频创意总监。你的任务是根据输入的人物图片和文字描述，生成同步的数字人视频组件。

重要原则：
- 生成结果中的所有字段语言必须跟随用户输入的语言。如果用户使用中文输入，所有输出字段（i2i_prompt、tts_text、voice_style）都必须使用中文；如果用户使用英文输入，则所有输出字段都使用英文。

对于 i2i_prompt（图生图修改提示词）：
- 强调"改"和"一致性"，不需要描述图片本身的内容
- 明确说明如何修改输入图片中的人物：例如"将输入图片中的人物改为穿着性感纯欲的睡衣坐在床上，保持人物特征的一致性（面部特征、身材等）"
- 描述需要改变的服装、场景、姿势、表情等
- 强调保持人物特征的一致性（面部特征、身材、年龄等）
- 确保修改后的描述与脚本内容相匹配

对于 tts_text（语音脚本）：
- 根据用户的输入描述生成相应的口语脚本
- 脚本内容应与用户的描述和场景相匹配
- 使用自然、对话式的语言
- 长度应适合数字人视频使用

对于 voice_style（语调指令）：
- 根据用户的输入描述和场景特点，提供相应的配音指导
- 描述配音应该体现的风格、情感、语气等特点
- 确保语调指令与脚本内容相匹配`,
            customOutputs: [
              { id: 'i2i_prompt', label: '场景修改提示词', description: '图生图修改提示词。强调"改"和"一致性"，明确说明如何修改输入图片中的人物。' },
              { id: 'tts_text', label: '数字人脚本', description: '根据用户描述生成的口语脚本，与场景相匹配。' },
              { id: 'voice_style', label: '语调', description: '根据用户描述和场景特点生成的配音指导，描述应该体现的风格、情感、语气等。' }
            ]
        } },
        { id: 'node-i2i-gen', toolId: 'image-to-image', x: 900, y: 50, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-voice', toolId: 'tts', x: 900, y: 450, status: NodeStatus.IDLE, data: { model: 'lightx2v', voiceType: 'zh_female_vv_uranus_bigtts', resourceId: 'seed-tts-2.0' } },
        { id: 'node-final-avatar', toolId: 'avatar-gen', x: 1500, y: 250, status: NodeStatus.IDLE, data: {} }
      ]
    },
    {
      id: 'preset-storyboard-9',
      name: '9分镜故事板视频工作流',
      updatedAt: Date.now(),
      isDirty: false,
      isRunning: false,
      env: {
        lightx2v_url: "",
        lightx2v_token: ""
      },
      globalInputs: {},
      history: [],
      showIntermediateResults: true,
      connections: [
        // Input to first image and planner
        { id: 'c1', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-1', targetPortId: 'in-image' },
        { id: 'c2', sourceNodeId: 'node-desc', sourcePortId: 'out-text', targetNodeId: 'node-planner', targetPortId: 'in-text' },
        // Planner to image edits (sequential generation based on previous image + character)
        { id: 'c3', sourceNodeId: 'node-planner', sourcePortId: 'scene1_prompt', targetNodeId: 'node-i2i-1', targetPortId: 'in-text' },
        { id: 'c4', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-2', targetPortId: 'in-image' },
        { id: 'c5', sourceNodeId: 'node-i2i-1', sourcePortId: 'out-image', targetNodeId: 'node-i2i-2', targetPortId: 'in-image' },
        { id: 'c6', sourceNodeId: 'node-planner', sourcePortId: 'scene2_prompt', targetNodeId: 'node-i2i-2', targetPortId: 'in-text' },
        { id: 'c7', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-3', targetPortId: 'in-image' },
        { id: 'c8', sourceNodeId: 'node-i2i-2', sourcePortId: 'out-image', targetNodeId: 'node-i2i-3', targetPortId: 'in-image' },
        { id: 'c9', sourceNodeId: 'node-planner', sourcePortId: 'scene3_prompt', targetNodeId: 'node-i2i-3', targetPortId: 'in-text' },
        { id: 'c10', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-4', targetPortId: 'in-image' },
        { id: 'c11', sourceNodeId: 'node-i2i-3', sourcePortId: 'out-image', targetNodeId: 'node-i2i-4', targetPortId: 'in-image' },
        { id: 'c12', sourceNodeId: 'node-planner', sourcePortId: 'scene4_prompt', targetNodeId: 'node-i2i-4', targetPortId: 'in-text' },
        { id: 'c13', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-5', targetPortId: 'in-image' },
        { id: 'c14', sourceNodeId: 'node-i2i-4', sourcePortId: 'out-image', targetNodeId: 'node-i2i-5', targetPortId: 'in-image' },
        { id: 'c15', sourceNodeId: 'node-planner', sourcePortId: 'scene5_prompt', targetNodeId: 'node-i2i-5', targetPortId: 'in-text' },
        { id: 'c16', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-6', targetPortId: 'in-image' },
        { id: 'c17', sourceNodeId: 'node-i2i-5', sourcePortId: 'out-image', targetNodeId: 'node-i2i-6', targetPortId: 'in-image' },
        { id: 'c18', sourceNodeId: 'node-planner', sourcePortId: 'scene6_prompt', targetNodeId: 'node-i2i-6', targetPortId: 'in-text' },
        { id: 'c19', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-7', targetPortId: 'in-image' },
        { id: 'c20', sourceNodeId: 'node-i2i-6', sourcePortId: 'out-image', targetNodeId: 'node-i2i-7', targetPortId: 'in-image' },
        { id: 'c21', sourceNodeId: 'node-planner', sourcePortId: 'scene7_prompt', targetNodeId: 'node-i2i-7', targetPortId: 'in-text' },
        { id: 'c22', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-8', targetPortId: 'in-image' },
        { id: 'c23', sourceNodeId: 'node-i2i-7', sourcePortId: 'out-image', targetNodeId: 'node-i2i-8', targetPortId: 'in-image' },
        { id: 'c24', sourceNodeId: 'node-planner', sourcePortId: 'scene8_prompt', targetNodeId: 'node-i2i-8', targetPortId: 'in-text' },
        { id: 'c25', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-9', targetPortId: 'in-image' },
        { id: 'c26', sourceNodeId: 'node-i2i-8', sourcePortId: 'out-image', targetNodeId: 'node-i2i-9', targetPortId: 'in-image' },
        { id: 'c27', sourceNodeId: 'node-planner', sourcePortId: 'scene9_prompt', targetNodeId: 'node-i2i-9', targetPortId: 'in-text' },
        // Image to video generation (i2v: each image generates one video)
        { id: 'c28', sourceNodeId: 'node-i2i-1', sourcePortId: 'out-image', targetNodeId: 'node-video-1', targetPortId: 'in-image' },
        { id: 'c29', sourceNodeId: 'node-planner', sourcePortId: 'scene1_video', targetNodeId: 'node-video-1', targetPortId: 'in-text' },
        { id: 'c30', sourceNodeId: 'node-i2i-2', sourcePortId: 'out-image', targetNodeId: 'node-video-2', targetPortId: 'in-image' },
        { id: 'c31', sourceNodeId: 'node-planner', sourcePortId: 'scene2_video', targetNodeId: 'node-video-2', targetPortId: 'in-text' },
        { id: 'c32', sourceNodeId: 'node-i2i-3', sourcePortId: 'out-image', targetNodeId: 'node-video-3', targetPortId: 'in-image' },
        { id: 'c33', sourceNodeId: 'node-planner', sourcePortId: 'scene3_video', targetNodeId: 'node-video-3', targetPortId: 'in-text' },
        { id: 'c34', sourceNodeId: 'node-i2i-4', sourcePortId: 'out-image', targetNodeId: 'node-video-4', targetPortId: 'in-image' },
        { id: 'c35', sourceNodeId: 'node-planner', sourcePortId: 'scene4_video', targetNodeId: 'node-video-4', targetPortId: 'in-text' },
        { id: 'c36', sourceNodeId: 'node-i2i-5', sourcePortId: 'out-image', targetNodeId: 'node-video-5', targetPortId: 'in-image' },
        { id: 'c37', sourceNodeId: 'node-planner', sourcePortId: 'scene5_video', targetNodeId: 'node-video-5', targetPortId: 'in-text' },
        { id: 'c38', sourceNodeId: 'node-i2i-6', sourcePortId: 'out-image', targetNodeId: 'node-video-6', targetPortId: 'in-image' },
        { id: 'c39', sourceNodeId: 'node-planner', sourcePortId: 'scene6_video', targetNodeId: 'node-video-6', targetPortId: 'in-text' },
        { id: 'c40', sourceNodeId: 'node-i2i-7', sourcePortId: 'out-image', targetNodeId: 'node-video-7', targetPortId: 'in-image' },
        { id: 'c41', sourceNodeId: 'node-planner', sourcePortId: 'scene7_video', targetNodeId: 'node-video-7', targetPortId: 'in-text' },
        { id: 'c42', sourceNodeId: 'node-i2i-8', sourcePortId: 'out-image', targetNodeId: 'node-video-8', targetPortId: 'in-image' },
        { id: 'c43', sourceNodeId: 'node-planner', sourcePortId: 'scene8_video', targetNodeId: 'node-video-8', targetPortId: 'in-text' },
        { id: 'c44', sourceNodeId: 'node-i2i-9', sourcePortId: 'out-image', targetNodeId: 'node-video-9', targetPortId: 'in-image' },
        { id: 'c45', sourceNodeId: 'node-planner', sourcePortId: 'scene9_video', targetNodeId: 'node-video-9', targetPortId: 'in-text' }
      ],
      nodes: [
        // Input nodes
        { id: 'node-char-img', toolId: 'image-input', x: 50, y: 500, status: NodeStatus.IDLE, data: { value: ['/assets/princess.png'] } },
        { id: 'node-desc', toolId: 'text-input', x: 50, y: 200, status: NodeStatus.IDLE, data: { value: "冰雪奇缘中的艾莎公主早晨醒来，在温馨的房间里梳妆打扮，然后望向窗外，窗外是很漂亮的阿伦黛尔小镇风光，然后镜头转向远景能够看到艾莎公主在窗边伸了个懒腰" } },
        // Planner node
        { id: 'node-planner', toolId: 'text-generation', x: 450, y: 350, status: NodeStatus.IDLE, data: { 
            model: 'doubao-seed-1-6-vision-250815',
            mode: 'custom',
            customInstruction: `你是一位专业的视频故事板规划师。你的任务是根据输入描述和人物图片，将其分解为恰好9个场景（分镜）用于顺序图像生成。
  
  关键：人物和场景一致性对这个故事板至关重要。
  
  重要原则：
  - 生成结果中的所有字段语言必须跟随用户输入的语言。如果用户使用中文输入，所有输出字段（sceneN_prompt、sceneN_video）都必须使用中文；如果用户使用英文输入，则所有输出字段都使用英文。
  
  图生图修改提示词原则（绝对关键）：
  - 强调"改"和"一致性"，不需要描述图片本身的内容
  - Scene 1: 将输入的人物图片中的人物改为描述的场景中的样子，保持人物特征的一致性（面部特征、身材、年龄等）
  - Scene 2及以后: 基于前一张场景图片，将其中的人物改为新场景中的样子，保持人物特征的一致性。如果人物在场景N改变了服装或发型，后续所有场景都要明确说明"保持与场景N相同的服装/发型"
  - 明确说明如何修改：例如"将输入图片（或前一张场景图片）中的人物改为在温馨的房间里梳妆打扮，保持人物特征的一致性"
  - 如果场景在同一地点，不要重复描述背景细节，只需说明"保持与前一场景相同的背景"
  - 描述需要改变的场景、姿势、表情等
  - 强调保持人物特征的一致性（面部特征、身材、年龄、体型等）
  - 不要描述光线、背景细节、氛围等图片细节
  
  人物一致性（绝对关键）：
  - 所有场景中必须出现相同的人物
  - 保持完全相同的人物特征：面部特征（眼睛、鼻子、嘴巴、脸型）、身材、年龄、体型、独特特征
  - 发型一致性：如果人物在任何场景中改变了发型（例如从凌乱到整齐，或改变颜色/样式），后续所有场景都必须保持相同的发型/颜色。一旦发型改变，必须在所有后续场景中保持一致。
  - 服装一致性：如果人物在任何场景中改变了服装（例如从睡衣到礼服，或从休闲到正式），后续所有场景都必须保持相同的服装。一旦服装改变，必须在所有后续场景中保持一致。在改变后的场景提示词中明确说明"人物穿着与场景N相同的服装"。
  - 人物的外观、年龄、体型和视觉风格必须在所有场景中保持完全相同
  
  场景一致性（绝对关键）：
  - 如果场景发生在同一地点（例如同一房间、同一户外区域），保持完全相同的背景元素、家具、道具、装饰、光线方向和空间布局
  - 如果两个连续场景在同一地点，描述完全相同的背景元素、家具布置和道具，避免视觉不一致
  - 保持相同的艺术风格（例如迪士尼风格、写实风格等）在所有场景中
  - 保持一致的光线方向、强度和配色方案在整个故事板中
  - 确保背景元素和设置在场景之间自然流动
  - 保持空间关系和视觉连续性
  
  图像生成方法：
  - Scene 1: 基于人物图片生成（建立人物的外观和第一个场景）
  - Scene 2: 基于Scene 1图片 + 人物图片生成（保持前一场景的连续性，同时保持人物身份）
  - Scene 3: 基于Scene 2图片 + 人物图片生成
  - 以此类推... 每个场景使用前一场景图片 + 人物图片以保持连续性和人物一致性
  
  视频生成：
  - 每个场景将使用图生视频（i2v）生成
  - sceneN_video应该描述该场景的运动/相机运动
  
  对于每个场景，输出：
  - sceneN_prompt: 强调"改"和"一致性"的图生图修改提示词，明确说明如何修改输入图片（Scene 1）或前一场景图片（Scene 2+）中的人物，保持人物一致性（特别是服装/发型如果改变 - 明确说明是否穿着与前一场景相同的服装/发型）和场景一致性（如果在同一地点 - 说明保持相同的背景元素）
  - sceneN_video: 描述相机运动和动作的视频运动提示词
  
  输出格式：JSON，包含以下字段：
  - scene1_prompt, scene1_video
  - scene2_prompt, scene2_video  
  - scene3_prompt, scene3_video
  - scene4_prompt, scene4_video
  - scene5_prompt, scene5_video
  - scene6_prompt, scene6_video
  - scene7_prompt, scene7_video
  - scene8_prompt, scene8_video
  - scene9_prompt, scene9_video
  
  重要：在每个提示词中，明确保持一致性：
  - 如果人物在场景N改变了服装，在scene(N+1)_prompt、scene(N+2)_prompt等中说明"人物穿着与场景N相同的[服装描述]"
  - 如果人物在场景N改变了发型，在后续提示词中说明"人物保持与场景N相同的发型/发色"
  - 如果场景共享同一地点，明确说明"保持与前一场景相同的背景元素、家具、道具和光线"`,
            customOutputs: [
              { id: 'scene1_prompt', label: 'Scene 1 Image Prompt', description: 'Detailed image prompt for scene 1 with character' },
              { id: 'scene1_video', label: 'Scene 1 Video Prompt', description: 'Video motion prompt for scene 1' },
              { id: 'scene2_prompt', label: 'Scene 2 Image Prompt', description: 'Detailed image prompt for scene 2 with character' },
              { id: 'scene2_video', label: 'Scene 2 Video Prompt', description: 'Video motion prompt for scene 2' },
              { id: 'scene3_prompt', label: 'Scene 3 Image Prompt', description: 'Detailed image prompt for scene 3 with character' },
              { id: 'scene3_video', label: 'Scene 3 Video Prompt', description: 'Video motion prompt for scene 3' },
              { id: 'scene4_prompt', label: 'Scene 4 Image Prompt', description: 'Detailed image prompt for scene 4 with character' },
              { id: 'scene4_video', label: 'Scene 4 Video Prompt', description: 'Video motion prompt for scene 4' },
              { id: 'scene5_prompt', label: 'Scene 5 Image Prompt', description: 'Detailed image prompt for scene 5 with character' },
              { id: 'scene5_video', label: 'Scene 5 Video Prompt', description: 'Video motion prompt for scene 5' },
              { id: 'scene6_prompt', label: 'Scene 6 Image Prompt', description: 'Detailed image prompt for scene 6 with character' },
              { id: 'scene6_video', label: 'Scene 6 Video Prompt', description: 'Video motion prompt for scene 6' },
              { id: 'scene7_prompt', label: 'Scene 7 Prompt', description: 'Image/video prompt for scene 7' },
              { id: 'scene7_video', label: 'Scene 7 Video Prompt', description: 'Video motion prompt for scene 7' },
              { id: 'scene8_prompt', label: 'Scene 8 Prompt', description: 'Image/video prompt for scene 8' },
              { id: 'scene8_video', label: 'Scene 8 Video Prompt', description: 'Video motion prompt for scene 8' },
              { id: 'scene9_prompt', label: 'Scene 9 Prompt', description: 'Image/video prompt for scene 9' },
              { id: 'scene9_video', label: 'Scene 9 Video Prompt', description: 'Video motion prompt for scene 9' }
            ]
        } },
        // Image-to-image nodes for all 9 scenes (sequential generation based on previous image + character)
        { id: 'node-i2i-1', toolId: 'image-to-image', x: 900, y: 50, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-i2i-2', toolId: 'image-to-image', x: 900, y: 150, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-i2i-3', toolId: 'image-to-image', x: 900, y: 250, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-i2i-4', toolId: 'image-to-image', x: 900, y: 350, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-i2i-5', toolId: 'image-to-image', x: 900, y: 450, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-i2i-6', toolId: 'image-to-image', x: 900, y: 550, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-i2i-7', toolId: 'image-to-image', x: 900, y: 650, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-i2i-8', toolId: 'image-to-image', x: 900, y: 750, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-i2i-9', toolId: 'image-to-image', x: 900, y: 850, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        // Video generation nodes (all 9 scenes using image-to-video)
        { id: 'node-video-1', toolId: 'video-gen-image', x: 1500, y: 50, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
        { id: 'node-video-2', toolId: 'video-gen-image', x: 1500, y: 150, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
        { id: 'node-video-3', toolId: 'video-gen-image', x: 1500, y: 250, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
        { id: 'node-video-4', toolId: 'video-gen-image', x: 1500, y: 350, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
        { id: 'node-video-5', toolId: 'video-gen-image', x: 1500, y: 450, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
        { id: 'node-video-6', toolId: 'video-gen-image', x: 1500, y: 550, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
        { id: 'node-video-7', toolId: 'video-gen-image', x: 1500, y: 650, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
        { id: 'node-video-8', toolId: 'video-gen-image', x: 1500, y: 750, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
        { id: 'node-video-9', toolId: 'video-gen-image', x: 1500, y: 850, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } }
      ]
    },
    {
      id: 'preset-multi-shot-singing',
      name: '数字人多机位唱歌',
      updatedAt: Date.now(),
      isDirty: false,
      isRunning: false,
      env: {
        lightx2v_url: "",
        lightx2v_token: ""
      },
      globalInputs: {},
      history: [],
      showIntermediateResults: true,
      connections: [
        // Input to AI Chat
        { id: 'c1', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-planner', targetPortId: 'in-image' },
        { id: 'c2', sourceNodeId: 'node-text-in', sourcePortId: 'out-text', targetNodeId: 'node-planner', targetPortId: 'in-text' },
        // AI Chat to Image-to-Image (9 shots)
        { id: 'c3', sourceNodeId: 'node-planner', sourcePortId: 'shot1_image_prompt', targetNodeId: 'node-i2i-1', targetPortId: 'in-text' },
        { id: 'c4', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-1', targetPortId: 'in-image' },
        { id: 'c5', sourceNodeId: 'node-planner', sourcePortId: 'shot2_image_prompt', targetNodeId: 'node-i2i-2', targetPortId: 'in-text' },
        { id: 'c6', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-2', targetPortId: 'in-image' },
        { id: 'c7', sourceNodeId: 'node-planner', sourcePortId: 'shot3_image_prompt', targetNodeId: 'node-i2i-3', targetPortId: 'in-text' },
        { id: 'c8', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-3', targetPortId: 'in-image' },
        { id: 'c9', sourceNodeId: 'node-planner', sourcePortId: 'shot4_image_prompt', targetNodeId: 'node-i2i-4', targetPortId: 'in-text' },
        { id: 'c10', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-4', targetPortId: 'in-image' },
        { id: 'c11', sourceNodeId: 'node-planner', sourcePortId: 'shot5_image_prompt', targetNodeId: 'node-i2i-5', targetPortId: 'in-text' },
        { id: 'c12', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-5', targetPortId: 'in-image' },
        { id: 'c28', sourceNodeId: 'node-planner', sourcePortId: 'shot6_image_prompt', targetNodeId: 'node-i2i-6', targetPortId: 'in-text' },
        { id: 'c29', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-6', targetPortId: 'in-image' },
        { id: 'c30', sourceNodeId: 'node-planner', sourcePortId: 'shot7_image_prompt', targetNodeId: 'node-i2i-7', targetPortId: 'in-text' },
        { id: 'c31', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-7', targetPortId: 'in-image' },
        { id: 'c32', sourceNodeId: 'node-planner', sourcePortId: 'shot8_image_prompt', targetNodeId: 'node-i2i-8', targetPortId: 'in-text' },
        { id: 'c33', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-8', targetPortId: 'in-image' },
        { id: 'c34', sourceNodeId: 'node-planner', sourcePortId: 'shot9_image_prompt', targetNodeId: 'node-i2i-9', targetPortId: 'in-text' },
        { id: 'c35', sourceNodeId: 'node-char-img', sourcePortId: 'out-image', targetNodeId: 'node-i2i-9', targetPortId: 'in-image' },
        // Image-to-Image to Video Generation (shots 1, 2, 5, 6, 8, 9 -> avatar-gen; shots 3, 4, 7 -> video-gen-image)
        { id: 'c13', sourceNodeId: 'node-i2i-1', sourcePortId: 'out-image', targetNodeId: 'node-avatar-1', targetPortId: 'in-image' },
        { id: 'c14', sourceNodeId: 'node-audio-in', sourcePortId: 'out-audio', targetNodeId: 'node-avatar-1', targetPortId: 'in-audio' },
        { id: 'c15', sourceNodeId: 'node-planner', sourcePortId: 'shot1_video_prompt', targetNodeId: 'node-avatar-1', targetPortId: 'in-text' },
        { id: 'c16', sourceNodeId: 'node-i2i-2', sourcePortId: 'out-image', targetNodeId: 'node-avatar-2', targetPortId: 'in-image' },
        { id: 'c17', sourceNodeId: 'node-audio-in', sourcePortId: 'out-audio', targetNodeId: 'node-avatar-2', targetPortId: 'in-audio' },
        { id: 'c18', sourceNodeId: 'node-planner', sourcePortId: 'shot2_video_prompt', targetNodeId: 'node-avatar-2', targetPortId: 'in-text' },
        // Shot 3 (Wide) - Image-to-Video
        { id: 'c19', sourceNodeId: 'node-i2i-3', sourcePortId: 'out-image', targetNodeId: 'node-video-3', targetPortId: 'in-image' },
        { id: 'c21', sourceNodeId: 'node-planner', sourcePortId: 'shot3_video_prompt', targetNodeId: 'node-video-3', targetPortId: 'in-text' },
        // Shot 4 (Top-down) - Image-to-Video
        { id: 'c22', sourceNodeId: 'node-i2i-4', sourcePortId: 'out-image', targetNodeId: 'node-video-4', targetPortId: 'in-image' },
        { id: 'c24', sourceNodeId: 'node-planner', sourcePortId: 'shot4_video_prompt', targetNodeId: 'node-video-4', targetPortId: 'in-text' },
        { id: 'c25', sourceNodeId: 'node-i2i-5', sourcePortId: 'out-image', targetNodeId: 'node-avatar-5', targetPortId: 'in-image' },
        { id: 'c26', sourceNodeId: 'node-audio-in', sourcePortId: 'out-audio', targetNodeId: 'node-avatar-5', targetPortId: 'in-audio' },
        { id: 'c27', sourceNodeId: 'node-planner', sourcePortId: 'shot5_video_prompt', targetNodeId: 'node-avatar-5', targetPortId: 'in-text' },
        { id: 'c36', sourceNodeId: 'node-i2i-6', sourcePortId: 'out-image', targetNodeId: 'node-avatar-6', targetPortId: 'in-image' },
        { id: 'c37', sourceNodeId: 'node-audio-in', sourcePortId: 'out-audio', targetNodeId: 'node-avatar-6', targetPortId: 'in-audio' },
        { id: 'c38', sourceNodeId: 'node-planner', sourcePortId: 'shot6_video_prompt', targetNodeId: 'node-avatar-6', targetPortId: 'in-text' },
        // Shot 7 (Extreme wide) - Image-to-Video
        { id: 'c39', sourceNodeId: 'node-i2i-7', sourcePortId: 'out-image', targetNodeId: 'node-video-7', targetPortId: 'in-image' },
        { id: 'c41', sourceNodeId: 'node-planner', sourcePortId: 'shot7_video_prompt', targetNodeId: 'node-video-7', targetPortId: 'in-text' },
        // Shot 8 (Over-shoulder) - Image-to-Video
        { id: 'c42', sourceNodeId: 'node-i2i-8', sourcePortId: 'out-image', targetNodeId: 'node-video-8', targetPortId: 'in-image' },
        { id: 'c44', sourceNodeId: 'node-planner', sourcePortId: 'shot8_video_prompt', targetNodeId: 'node-video-8', targetPortId: 'in-text' },
        { id: 'c45', sourceNodeId: 'node-i2i-9', sourcePortId: 'out-image', targetNodeId: 'node-avatar-9', targetPortId: 'in-image' },
        { id: 'c46', sourceNodeId: 'node-audio-in', sourcePortId: 'out-audio', targetNodeId: 'node-avatar-9', targetPortId: 'in-audio' },
        { id: 'c47', sourceNodeId: 'node-planner', sourcePortId: 'shot9_video_prompt', targetNodeId: 'node-avatar-9', targetPortId: 'in-text' }
      ],
      nodes: [
        // Input nodes
        { id: 'node-char-img', toolId: 'image-input', x: 50, y: 500, status: NodeStatus.IDLE, data: { value: ['/assets/singing_princess.png'] } },
        { id: 'node-audio-in', toolId: 'audio-input', x: 50, y: 700, status: NodeStatus.IDLE, data: { value: '/assets/let_it_go_part.wav' } },
        { id: 'node-text-in', toolId: 'text-input', x: 50, y: 200, status: NodeStatus.IDLE, data: { value: "冰雪奇缘中的艾莎公主正在演唱《Let It Go》，动作优雅，表情充满自信和力量" } },
        // AI Chat Planner (Doubao Vision)
        { id: 'node-planner', toolId: 'text-generation', x: 450, y: 400, status: NodeStatus.IDLE, data: { 
            model: 'doubao-seed-1-6-vision-250815',
            mode: 'custom',
            customInstruction: `你是一位专业的音乐视频多机位导演。你的任务是根据输入的角色图片和可选的文字描述，为演唱表演生成9个不同机位的详细描述。
  
  关键：人物一致性至关重要。相同的人物必须在所有机位中出现，具有完全相同的面部特征、身材、服装和外观。
  
  重要原则：
  - 生成结果中的所有字段语言必须跟随用户输入的语言。如果用户使用中文输入，所有输出字段（shotN_image_prompt、shotN_video_prompt）都必须使用中文；如果用户使用英文输入，则所有输出字段都使用英文。
  
  图生图修改提示词原则（绝对关键）：
  - 强调"改"和"一致性"，不需要描述图片本身的内容
  - 明确说明如何修改输入图片中的人物：例如"将输入图片中的人物改为特写角度，保持人物特征的一致性（面部特征、身材、服装等）"
  - 描述需要改变的机位、角度、构图、姿势等
  - 强调保持人物特征的一致性（面部特征、身材、服装、外观等）
  - 不要描述光线、背景细节、氛围等图片细节
  
  生成9个不同的机位：
  1. Shot 1 - 特写: 以脸部为主的镜头，展示详细的面部表情、情感和口型同步
  2. Shot 2 - 中景: 上半身镜头，展示头部、肩膀和一些手臂动作。图生图提示词应强调人物的上半身姿势、手势和动态的手臂动作，匹配演唱表演。包括人物上半身服装的细节、肩膀位置以及手臂如何构成画面。
  3. Shot 3 - 全景: 全身镜头，展示整个人物和背景。图生图提示词应清楚描述全身姿势、腿部位置、整体肢体语言以及人物如何占据空间。包括完整服装、身体姿势以及人物与背景环境的关系。
  4. Shot 4 - 俯拍: 俯视或高角度镜头，从上往下看人物
  5. Shot 5 - 侧方位: 侧面或3/4角度镜头，从侧面展示人物
  6. Shot 6 - 正面半身: 正面中景镜头，从正面展示人物的上半身，捕捉脸部、肩膀和上躯干。图生图提示词应强调人物的正面外观、面部表情、上半身姿势以及人物如何直接面对镜头。
  7. Shot 7 - 极大景: 极宽镜头，在广阔环境中展示人物，强调规模和氛围
  8. Shot 8 - 过肩: 过肩镜头，从背后展示人物，营造亲密感
  9. Shot 9 - 极特写: 极特写，聚焦眼睛、嘴巴或特定面部特征
  
  对于每个机位，生成：
  - shotN_image_prompt: 强调"改"和"一致性"的图生图修改提示词。明确说明如何将输入图片中的人物改为该特定机位和构图，保持完全相同的人物一致性（相同的面部、服装、外观）。描述需要改变的机位角度、构图、姿势/表情。对于机位2、3和6，提供特别详细的身体位置、姿势和空间构图描述。
  - shotN_video_prompt: 对于机位1、2、5、6、9（数字人机位）：详细的数字人视频动作描述，包括演唱手势、头部动作、肢体语言、面部表情和与歌曲节奏和能量匹配的动作。对于机位3、4、7、8（图生视频机位）：详细的视频运动描述，包括相机运动技巧（推、拉、摇、移、跟、缩放等）、转场效果和场景动态，以创建机位之间的平滑过渡并与歌曲的节奏和能量匹配。运动描述应专注于相机运动和视觉转场，以在不同机位角度之间创建无缝连接。
  
  输出格式：JSON，包含以下字段：
  - shot1_image_prompt, shot1_video_prompt
  - shot2_image_prompt, shot2_video_prompt
  - shot3_image_prompt, shot3_video_prompt
  - shot4_image_prompt, shot4_video_prompt
  - shot5_image_prompt, shot5_video_prompt
  - shot6_image_prompt, shot6_video_prompt
  - shot7_image_prompt, shot7_video_prompt
  - shot8_image_prompt, shot8_video_prompt
  - shot9_image_prompt, shot9_video_prompt
  
  重要： 
  - 在所有机位中保持完全相同的人物一致性（相同的面部、服装、外观）
  - 每个机位应该有独特的相机角度和构图
  - 对于数字人机位（1、2、5、6、9）：视频提示词应描述自然的演唱动作和表情
  - 对于图生视频机位（3、4、7、8）：视频提示词必须包括详细的相机运动技巧（推、拉、摇、移、跟、缩放等）和转场效果，以创建机位之间的平滑视觉过渡。这些运动描述对于无缝机位过渡至关重要，应与歌曲的节奏和能量匹配`,
            customOutputs: [
              { id: 'shot1_image_prompt', label: 'Shot 1 Image Prompt (Close-up)', description: 'Close-up shot image description' },
              { id: 'shot1_video_prompt', label: 'Shot 1 Video Prompt', description: 'Close-up shot video action description' },
              { id: 'shot2_image_prompt', label: 'Shot 2 Image Prompt (Medium)', description: 'Medium shot image description with emphasis on upper body posture, hand gestures, and arm movements' },
              { id: 'shot2_video_prompt', label: 'Shot 2 Video Prompt', description: 'Medium shot video action description' },
              { id: 'shot3_image_prompt', label: 'Shot 3 Image Prompt (Wide)', description: 'Wide shot image description with emphasis on full body pose, leg positioning, and spatial composition' },
              { id: 'shot3_video_prompt', label: 'Shot 3 Video Prompt', description: 'Wide shot video motion description with camera movement and transition effects' },
              { id: 'shot4_image_prompt', label: 'Shot 4 Image Prompt (Top-down)', description: 'Top-down shot image description' },
              { id: 'shot4_video_prompt', label: 'Shot 4 Video Prompt', description: 'Top-down shot video motion description with camera movement and transition effects' },
              { id: 'shot5_image_prompt', label: 'Shot 5 Image Prompt (Side)', description: 'Side angle shot image description' },
              { id: 'shot5_video_prompt', label: 'Shot 5 Video Prompt', description: 'Side angle shot video action description' },
              { id: 'shot6_image_prompt', label: 'Shot 6 Image Prompt (Front medium)', description: 'Front medium shot image description with emphasis on frontal appearance, facial expression, and upper body posture' },
              { id: 'shot6_video_prompt', label: 'Shot 6 Video Prompt', description: 'Low-angle shot video action description' },
              { id: 'shot7_image_prompt', label: 'Shot 7 Image Prompt (Extreme wide)', description: 'Extreme wide shot image description' },
              { id: 'shot7_video_prompt', label: 'Shot 7 Video Prompt', description: 'Extreme wide shot video motion description with camera movement and transition effects' },
              { id: 'shot8_image_prompt', label: 'Shot 8 Image Prompt (Over-shoulder)', description: 'Over-shoulder shot image description' },
              { id: 'shot8_video_prompt', label: 'Shot 8 Video Prompt', description: 'Over-shoulder shot video motion description with camera movement and transition effects' },
              { id: 'shot9_image_prompt', label: 'Shot 9 Image Prompt (Extreme close-up)', description: 'Extreme close-up shot image description' },
              { id: 'shot9_video_prompt', label: 'Shot 9 Video Prompt', description: 'Extreme close-up shot video action description' }
            ]
        } },
        // Image-to-Image nodes (9 shots)
        { id: 'node-i2i-1', toolId: 'image-to-image', x: 900, y: 50, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-i2i-2', toolId: 'image-to-image', x: 900, y: 150, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-i2i-3', toolId: 'image-to-image', x: 900, y: 250, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-i2i-4', toolId: 'image-to-image', x: 900, y: 350, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-i2i-5', toolId: 'image-to-image', x: 900, y: 450, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-i2i-6', toolId: 'image-to-image', x: 900, y: 550, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-i2i-7', toolId: 'image-to-image', x: 900, y: 650, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-i2i-8', toolId: 'image-to-image', x: 900, y: 750, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'node-i2i-9', toolId: 'image-to-image', x: 900, y: 850, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        // Avatar Video nodes (shots 1, 2, 5, 6, 9 - digital avatar)
        { id: 'node-avatar-1', toolId: 'avatar-gen', x: 1500, y: 50, status: NodeStatus.IDLE, data: {} },
        { id: 'node-avatar-2', toolId: 'avatar-gen', x: 1500, y: 150, status: NodeStatus.IDLE, data: {} },
        { id: 'node-avatar-5', toolId: 'avatar-gen', x: 1500, y: 450, status: NodeStatus.IDLE, data: {} },
        { id: 'node-avatar-6', toolId: 'avatar-gen', x: 1500, y: 550, status: NodeStatus.IDLE, data: {} },
        { id: 'node-avatar-9', toolId: 'avatar-gen', x: 1500, y: 850, status: NodeStatus.IDLE, data: {} },
        // Image-to-Video nodes (shots 3, 4, 7, 8 - image-to-video)
        { id: 'node-video-3', toolId: 'video-gen-image', x: 1500, y: 250, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
        { id: 'node-video-4', toolId: 'video-gen-image', x: 1500, y: 350, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
        { id: 'node-video-7', toolId: 'video-gen-image', x: 1500, y: 650, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } },
        { id: 'node-video-8', toolId: 'video-gen-image', x: 1500, y: 750, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: '9:16' } }
      ]
    },
    {
      id: 'preset-chibi-avatar',
      name: 'Q版数字人',
      updatedAt: Date.now(),
      isDirty: false,
      isRunning: false,
      env: {
        lightx2v_url: "",
        lightx2v_token: ""
      },
      globalInputs: {},
      history: [],
      showIntermediateResults: true,
      connections: [
        { id: 'c1', sourceNodeId: 'chibi-node-person', sourcePortId: 'out-image', targetNodeId: 'chibi-node-ai', targetPortId: 'in-image' },
        { id: 'c2', sourceNodeId: 'chibi-node-input', sourcePortId: 'out-text', targetNodeId: 'chibi-node-ai', targetPortId: 'in-text' },
        { id: 'c3', sourceNodeId: 'chibi-node-ai', sourcePortId: 'i2i_prompt', targetNodeId: 'chibi-node-i2i', targetPortId: 'in-text' },
        { id: 'c4', sourceNodeId: 'chibi-node-person', sourcePortId: 'out-image', targetNodeId: 'chibi-node-i2i', targetPortId: 'in-image' },
        { id: 'c5', sourceNodeId: 'chibi-node-ai', sourcePortId: 'tts_text', targetNodeId: 'chibi-node-tts', targetPortId: 'in-text' },
        { id: 'c6', sourceNodeId: 'chibi-node-ai', sourcePortId: 'voice_style', targetNodeId: 'chibi-node-tts', targetPortId: 'in-context-tone' },
        { id: 'c7', sourceNodeId: 'chibi-node-i2i', sourcePortId: 'out-image', targetNodeId: 'chibi-node-avatar', targetPortId: 'in-image' },
        { id: 'c8', sourceNodeId: 'chibi-node-tts', sourcePortId: 'out-audio', targetNodeId: 'chibi-node-avatar', targetPortId: 'in-audio' },
        { id: 'c9', sourceNodeId: 'chibi-node-ai', sourcePortId: 's2v_prompt', targetNodeId: 'chibi-node-avatar', targetPortId: 'in-text' }
      ],
      nodes: [
        { id: 'chibi-node-person', toolId: 'image-input', x: 50, y: 200, status: NodeStatus.IDLE, data: { value: [] } },
        { id: 'chibi-node-doll-ref', toolId: 'image-input', x: 50, y: 50, status: NodeStatus.IDLE, data: { value: ['/assets/doll.jpg'] } },
        { id: 'chibi-node-input', toolId: 'text-input', x: 50, y: 400, status: NodeStatus.IDLE, data: { value: "角色说的话和角色背景描述" } },
        { id: 'chibi-node-ai', toolId: 'text-generation', x: 450, y: 300, status: NodeStatus.IDLE, data: { 
            model: 'doubao-seed-1-6-vision-250815',
            mode: 'custom',
            customInstruction: `你是一位专业的Q版数字人视频创意总监。你的任务是根据输入的人物图片和用户描述（角色说的话和角色背景），生成完整的Q版数字人视频组件。

重要原则：
- 生成结果中的所有字段语言必须跟随用户输入的语言。如果用户使用中文输入，所有输出字段（i2i_prompt、tts_text、voice_style、s2v_prompt）都必须使用中文；如果用户使用英文输入，则所有输出字段都使用英文。

参考风格：Q版球关节娃娃风格，将照片中的人物转换为Q版球关节娃娃，保持人物的面部特征一致但风格化，大脑袋小身体比例，光滑的树脂质感皮肤，闪亮的娃娃眼睛，简化的面部结构，柔和的腮红，可见的关节段，可爱的艺术玩具美学，角色自然站立，双脚着地，干净的纯白背景，柔和均匀的光线，高细节，无阴影杂乱。

对于 i2i_prompt（图生图修改提示词）：
- 基于输入的人物图片，将其转换为Q版球关节娃娃风格
- 强调"改"和"一致性"，不需要描述图片本身的内容
- 明确说明如何修改：将输入图片中的人物转换为Q版球关节娃娃风格，保持人物面部特征的一致性（面部特征、表情等），但风格化为大脑袋小身体比例，光滑的树脂质感皮肤，闪亮的娃娃眼睛，简化的面部结构，柔和的腮红，可见的关节段，可爱的艺术玩具美学
- 角色自然站立，双脚着地，干净的纯白背景，柔和均匀的光线，高细节，无阴影杂乱
- 确保修改后的描述与角色背景相匹配

对于 tts_text（语音脚本）：
- 根据用户的输入描述生成相应的口语脚本
- 脚本内容应与用户的描述和角色背景相匹配
- 使用自然、对话式的语言
- 长度应适合数字人视频使用（正常语速下约20-40秒）

对于 voice_style（语调指令）：
- 根据用户的输入描述和角色背景特点，提供相应的配音指导
- 描述配音应该体现的风格、情感、语气等特点
- 确保语调指令与脚本内容和角色背景相匹配
- Q版角色通常需要可爱、活泼、轻快的语调

对于 s2v_prompt（数字人视频动作提示）：
- 描述自然、真实的说话手势和动作
- 包含头部动作（点头、倾斜、轻微转动）
- 描述与语音内容和情感语调匹配的面部表情
- Q版角色可以有一些可爱的动作，如轻微摆动、活泼的手势
- 指定眼神接触和视线方向
- 确保动作与语音节奏同步
- 足够详细，以指导数字人视频生成自然、逼真的效果`,
            customOutputs: [
              { id: 'i2i_prompt', label: '改图提示词', description: '将人物转换为Q版球关节娃娃风格的图生图修改提示词。' },
              { id: 'tts_text', label: 'TTS文案', description: '根据用户描述生成的语音脚本。' },
              { id: 'voice_style', label: 'TTS语气指令', description: '根据用户描述和角色背景生成的配音指导。' },
              { id: 's2v_prompt', label: 'S2V提示词', description: '数字人视频动作和运动的描述。' }
            ]
        } },
        { id: 'chibi-node-i2i', toolId: 'image-to-image', x: 900, y: 50, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: '9:16' } },
        { id: 'chibi-node-tts', toolId: 'tts', x: 900, y: 450, status: NodeStatus.IDLE, data: { model: 'lightx2v', voiceType: 'zh_female_vv_uranus_bigtts', resourceId: 'seed-tts-2.0' } },
        { id: 'chibi-node-avatar', toolId: 'avatar-gen', x: 1500, y: 250, status: NodeStatus.IDLE, data: { model: 'SekoTalk' } }
      ]
    },
    {
      id: 'preset-cinematic-oner',
      name: '大师级运镜一镜到底首尾帧视频工作流',
      updatedAt: Date.now(),
      isDirty: false,
      isRunning: false,
      env: {
        lightx2v_url: "",
        lightx2v_token: ""
      },
      globalInputs: {},
      history: [],
      showIntermediateResults: true,
      connections: [
        // Input to planner
        { id: 'oner-c1', sourceNodeId: 'oner-node-desc', sourcePortId: 'out-text', targetNodeId: 'oner-node-planner', targetPortId: 'in-text' },
        // Planner to image generation (5 cinematic shots)
        { id: 'oner-c2', sourceNodeId: 'oner-node-planner', sourcePortId: 'shot1_image_prompt', targetNodeId: 'oner-node-img-1', targetPortId: 'in-text' },
        { id: 'oner-c3', sourceNodeId: 'oner-node-img-1', sourcePortId: 'out-image', targetNodeId: 'oner-node-img-2', targetPortId: 'in-image' },
        { id: 'oner-c4', sourceNodeId: 'oner-node-planner', sourcePortId: 'shot2_image_prompt', targetNodeId: 'oner-node-img-2', targetPortId: 'in-text' },
        { id: 'oner-c5', sourceNodeId: 'oner-node-img-2', sourcePortId: 'out-image', targetNodeId: 'oner-node-img-3', targetPortId: 'in-image' },
        { id: 'oner-c6', sourceNodeId: 'oner-node-planner', sourcePortId: 'shot3_image_prompt', targetNodeId: 'oner-node-img-3', targetPortId: 'in-text' },
        { id: 'oner-c7', sourceNodeId: 'oner-node-img-3', sourcePortId: 'out-image', targetNodeId: 'oner-node-img-4', targetPortId: 'in-image' },
        { id: 'oner-c8', sourceNodeId: 'oner-node-planner', sourcePortId: 'shot4_image_prompt', targetNodeId: 'oner-node-img-4', targetPortId: 'in-text' },
        { id: 'oner-c9', sourceNodeId: 'oner-node-img-4', sourcePortId: 'out-image', targetNodeId: 'oner-node-img-5', targetPortId: 'in-image' },
        { id: 'oner-c10', sourceNodeId: 'oner-node-planner', sourcePortId: 'shot5_image_prompt', targetNodeId: 'oner-node-img-5', targetPortId: 'in-text' },
        // Image to video generation (dual-frame: start frame + end frame)
        // Video 1: shot1 (start) -> shot2 (end)
        { id: 'oner-c11', sourceNodeId: 'oner-node-img-1', sourcePortId: 'out-image', targetNodeId: 'oner-node-video-1', targetPortId: 'in-image-start' },
        { id: 'oner-c11b', sourceNodeId: 'oner-node-img-2', sourcePortId: 'out-image', targetNodeId: 'oner-node-video-1', targetPortId: 'in-image-end' },
        { id: 'oner-c12', sourceNodeId: 'oner-node-planner', sourcePortId: 'shot1_video_motion', targetNodeId: 'oner-node-video-1', targetPortId: 'in-text' },
        // Video 2: shot2 (start) -> shot3 (end)
        { id: 'oner-c13', sourceNodeId: 'oner-node-img-2', sourcePortId: 'out-image', targetNodeId: 'oner-node-video-2', targetPortId: 'in-image-start' },
        { id: 'oner-c13b', sourceNodeId: 'oner-node-img-3', sourcePortId: 'out-image', targetNodeId: 'oner-node-video-2', targetPortId: 'in-image-end' },
        { id: 'oner-c14', sourceNodeId: 'oner-node-planner', sourcePortId: 'shot2_video_motion', targetNodeId: 'oner-node-video-2', targetPortId: 'in-text' },
        // Video 3: shot3 (start) -> shot4 (end)
        { id: 'oner-c15', sourceNodeId: 'oner-node-img-3', sourcePortId: 'out-image', targetNodeId: 'oner-node-video-3', targetPortId: 'in-image-start' },
        { id: 'oner-c15b', sourceNodeId: 'oner-node-img-4', sourcePortId: 'out-image', targetNodeId: 'oner-node-video-3', targetPortId: 'in-image-end' },
        { id: 'oner-c16', sourceNodeId: 'oner-node-planner', sourcePortId: 'shot3_video_motion', targetNodeId: 'oner-node-video-3', targetPortId: 'in-text' },
        // Video 4: shot4 (start) -> shot5 (end)
        { id: 'oner-c17', sourceNodeId: 'oner-node-img-4', sourcePortId: 'out-image', targetNodeId: 'oner-node-video-4', targetPortId: 'in-image-start' },
        { id: 'oner-c17b', sourceNodeId: 'oner-node-img-5', sourcePortId: 'out-image', targetNodeId: 'oner-node-video-4', targetPortId: 'in-image-end' },
        { id: 'oner-c18', sourceNodeId: 'oner-node-planner', sourcePortId: 'shot4_video_motion', targetNodeId: 'oner-node-video-4', targetPortId: 'in-text' },
        // Video 5: shot5 (start) -> shot5 (end, same frame for final shot)
        { id: 'oner-c19', sourceNodeId: 'oner-node-img-5', sourcePortId: 'out-image', targetNodeId: 'oner-node-video-5', targetPortId: 'in-image-start' },
        { id: 'oner-c19b', sourceNodeId: 'oner-node-img-5', sourcePortId: 'out-image', targetNodeId: 'oner-node-video-5', targetPortId: 'in-image-end' },
        { id: 'oner-c20', sourceNodeId: 'oner-node-planner', sourcePortId: 'shot5_video_motion', targetNodeId: 'oner-node-video-5', targetPortId: 'in-text' }
      ],
      nodes: [
        { id: 'oner-node-desc', toolId: 'text-input', x: 50, y: 400, status: NodeStatus.IDLE, data: { value: "一座未来主义赛博朋克城市的宏大全景，从高空俯瞰整座城市，镜头逐渐下降穿过云雾，掠过摩天大楼的玻璃幕墙，最终聚焦到繁华街道上的人群和霓虹灯" } },
        { id: 'oner-node-planner', toolId: 'text-generation', x: 450, y: 400, status: NodeStatus.IDLE, data: { 
            model: 'deepseek-v3-2-251201',
            mode: 'custom',
            customInstruction: `你是一位专业的电影级视频分镜设计师。你的任务是根据用户的场景描述，设计一个"一镜到底"的连续运镜视频。
  
  重要原则：
  - 生成结果中的所有字段语言必须跟随用户输入的语言。如果用户使用中文输入，所有输出字段（shotN_image_prompt、shotN_video_motion）都必须使用中文；如果用户使用英文输入，则所有输出字段都使用英文。
  - 这是"一镜到底"视频，所有镜头必须形成连续、流畅的视觉过渡
  - 运用多种电影级运镜技巧：极远景、全景、中景、近景、特写、推拉镜头、跟随镜头、环绕镜头、升降镜头、俯仰镜头等
  - 场景必须宏大、震撼，具有视觉冲击力
  - 保持场景一致性和视觉连贯性（同一场景的不同视角）
  - 每个分镜的视觉风格、色调、氛围必须统一
  - 镜头运动要流畅自然，前后分镜要有逻辑连接
  
  图生图修改提示词原则（对于分镜2-5）：
  - 强调"改"和"一致性"，不需要描述图片本身的内容
  - Shot 1: 使用文生图，详细描述画面内容、构图、光线、氛围
  - Shot 2-5: 明确说明如何修改前一张分镜图片：例如"将前一张分镜图片改为中景构图，镜头逐渐接近，保持场景元素的一致性"
  - 描述需要改变的构图、视角、镜头位置等
  - 强调保持场景元素的一致性（如果是同一个场景，要说明"保持与前一分镜相同的场景元素"）
  - 不要重复描述光线、背景细节、氛围等图片细节
  
  输出要求：
  - 生成5个分镜，每个分镜包含：
    - shotN_image_prompt: Shot 1使用文生图提示词，详细描述画面内容、构图（构图规则，如三分法、对称等）、光线（自然光、人工光、色调、明暗对比）、氛围和情绪。Shot 2-5使用强调"改"和"一致性"的图生图修改提示词，明确说明如何修改前一张分镜图片，保持场景元素的连贯性。
    - shotN_video_motion: 详细的视频运动提示词，描述该分镜的相机运动、运镜方式、运动方向、速度
  
  分镜设计思路（参考）：
  - 分镜1: 极远景/全景 - 建立宏大场景，可能是高空俯瞰或全景展示（文生图）
  - 分镜2: 中景/推拉 - 镜头逐渐接近，或从一侧移动到另一侧（图生图：基于分镜1）
  - 分镜3: 中景/跟随/环绕 - 镜头运动，展现场景的深度和细节（图生图：基于分镜2）
  - 分镜4: 近景/特写 - 聚焦到场景中的关键元素或细节（图生图：基于分镜3）
  - 分镜5: 全景/极远景收尾 - 回到宏大的视角，形成视觉闭环（图生图：基于分镜4）
  
  每个video_motion应该（关键：这是首尾帧生视频，强调从当前镜头到下一个镜头的运镜切换）：
  - 使用电影式运镜语言，描述从当前分镜（首帧）到下一个分镜（尾帧）的相机运动
  - 重点描述运镜切换：如何从当前镜头的构图、角度、位置，通过相机运动过渡到下一个镜头的构图、角度、位置
  - 运用电影级运镜术语：
    * 推拉镜头（Dolly In/Out）：相机前后移动
    * 摇镜头（Pan）：相机左右旋转
    * 移镜头（Truck）：相机左右平移
    * 升降镜头（Crane/Boom）：相机上下移动
    * 跟随镜头（Follow/Tracking）：相机跟随主体移动
    * 环绕镜头（Orbit/Arc）：相机围绕主体旋转
    * 变焦（Zoom In/Out）：镜头焦距变化
    * 俯仰镜头（Tilt）：相机上下旋转
  - 描述运动速度（慢速、中速、快速）和节奏感
  - 描述运动轨迹和路径（直线、弧线、曲线等）
  - 强调视觉连贯性：确保从首帧到尾帧的过渡自然流畅，形成"一镜到底"的连续感
  - 对于最后一个分镜（shot5），可以描述一个收尾性的运镜，如缓慢拉远或环绕收尾
  
  输出JSON格式，包含以下字段：
  - shot1_image_prompt, shot1_video_motion
  - shot2_image_prompt, shot2_video_motion
  - shot3_image_prompt, shot3_video_motion
  - shot4_image_prompt, shot4_video_motion
  - shot5_image_prompt, shot5_video_motion`,
            customOutputs: [
              { id: 'shot1_image_prompt', label: '分镜1图像提示', description: '第一分镜的详细图像生成提示词（远景/全景，建立宏大场景）' },
              { id: 'shot1_video_motion', label: '分镜1运镜描述', description: '从分镜1到分镜2的运镜切换提示词，使用电影式运镜语言描述相机运动' },
              { id: 'shot2_image_prompt', label: '分镜2图像提示', description: '第二分镜的详细图像生成提示词（中景，镜头逐渐接近）' },
              { id: 'shot2_video_motion', label: '分镜2运镜描述', description: '从分镜2到分镜3的运镜切换提示词，使用电影式运镜语言描述相机运动' },
              { id: 'shot3_image_prompt', label: '分镜3图像提示', description: '第三分镜的详细图像生成提示词（中景/跟随，展现深度）' },
              { id: 'shot3_video_motion', label: '分镜3运镜描述', description: '从分镜3到分镜4的运镜切换提示词，使用电影式运镜语言描述相机运动' },
              { id: 'shot4_image_prompt', label: '分镜4图像提示', description: '第四分镜的详细图像生成提示词（近景/特写，聚焦细节）' },
              { id: 'shot4_video_motion', label: '分镜4运镜描述', description: '从分镜4到分镜5的运镜切换提示词，使用电影式运镜语言描述相机运动' },
              { id: 'shot5_image_prompt', label: '分镜5图像提示', description: '第五分镜的详细图像生成提示词（全景收尾，视觉闭环）' },
              { id: 'shot5_video_motion', label: '分镜5运镜描述', description: '分镜5的收尾运镜提示词，使用电影式运镜语言描述收尾性的相机运动' }
            ]
        } },
        { id: 'oner-node-img-1', toolId: 'text-to-image', x: 900, y: 100, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-2512', aspectRatio: "16:9" } },
        { id: 'oner-node-img-2', toolId: 'image-to-image', x: 900, y: 250, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: "16:9" } },
        { id: 'oner-node-img-3', toolId: 'image-to-image', x: 900, y: 400, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: "16:9" } },
        { id: 'oner-node-img-4', toolId: 'image-to-image', x: 900, y: 550, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: "16:9" } },
        { id: 'oner-node-img-5', toolId: 'image-to-image', x: 900, y: 700, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511', aspectRatio: "16:9" } },
        { id: 'oner-node-video-1', toolId: 'video-gen-dual-frame', x: 1500, y: 100, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: "16:9" } },
        { id: 'oner-node-video-2', toolId: 'video-gen-dual-frame', x: 1500, y: 250, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: "16:9" } },
        { id: 'oner-node-video-3', toolId: 'video-gen-dual-frame', x: 1500, y: 400, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: "16:9" } },
        { id: 'oner-node-video-4', toolId: 'video-gen-dual-frame', x: 1500, y: 550, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: "16:9" } },
        { id: 'oner-node-video-5', toolId: 'video-gen-dual-frame', x: 1500, y: 700, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: "16:9" } }
      ]
    },
    {
        id: 'preset-morph',
        name: '文生首尾帧视频工作流',
        updatedAt: Date.now(),
        isDirty: false,
        isRunning: false,
        env: {
          lightx2v_url: "",
          lightx2v_token: ""
        },
        globalInputs: {},
        history: [],
        showIntermediateResults: true,
        connections: [
          { id: 'c1', sourceNodeId: 'node-input', sourcePortId: 'out-text', targetNodeId: 'node-planner', targetPortId: 'in-text' },
          { id: 'c2', sourceNodeId: 'node-planner', sourcePortId: 'start_img_prompt', targetNodeId: 'node-start-frame', targetPortId: 'in-text' },
          { id: 'c3', sourceNodeId: 'node-start-frame', sourcePortId: 'out-image', targetNodeId: 'node-end-frame', targetPortId: 'in-image' },
          { id: 'c4', sourceNodeId: 'node-planner', sourcePortId: 'end_img_prompt', targetNodeId: 'node-end-frame', targetPortId: 'in-text' },
          { id: 'c5', sourceNodeId: 'node-start-frame', sourcePortId: 'out-image', targetNodeId: 'node-video', targetPortId: 'in-image-start' },
          { id: 'c6', sourceNodeId: 'node-end-frame', sourcePortId: 'out-image', targetNodeId: 'node-video', targetPortId: 'in-image-end' },
          { id: 'c7', sourceNodeId: 'node-planner', sourcePortId: 'video_motion_prompt', targetNodeId: 'node-video', targetPortId: 'in-text' }
        ],
        nodes: [
          { id: 'node-input', toolId: 'text-input', x: 50, y: 300, status: NodeStatus.IDLE, data: { value: "一座未来主义赛博朋克城市，从白天逐渐过渡到雨夜。" } },
          { id: 'node-planner', toolId: 'text-generation', x: 450, y: 300, status: NodeStatus.IDLE, data: { 
              model: 'deepseek-v3-2-251201',
              mode: 'custom',
              customInstruction: `You are a video planning assistant. Analyze the input description and generate detailed prompts for the start frame, end frame, and video motion.
  
  IMPORTANT: All output fields must use the same language as the user's input. If the user inputs in Chinese, all output fields (start_img_prompt, end_img_prompt, video_motion_prompt) must be in Chinese. If the user inputs in English, all output fields must be in English.
  
  Generate:
  - start_img_prompt: Detailed prompt for the initial image
  - end_img_prompt: Detailed prompt for the target image, based on the start
  - video_motion_prompt: Prompt describing the transition and camera motion`,
              customOutputs: [
                { id: 'start_img_prompt', label: 'Start Frame Prompt', description: 'Detailed prompt for the initial image.' },
                { id: 'end_img_prompt', label: 'End Frame Prompt', description: 'Detailed prompt for the target image, based on the start.' },
                { id: 'video_motion_prompt', label: 'Motion Prompt', description: 'Prompt describing the transition and camera motion.' }
              ]
          } },
          { id: 'node-start-frame', toolId: 'text-to-image', x: 900, y: 50, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-2512', aspectRatio: "16:9" } },
          { id: 'node-end-frame', toolId: 'image-to-image', x: 900, y: 550, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-Edit-2511' } },
          { id: 'node-video', toolId: 'video-gen-dual-frame', x: 1500, y: 300, status: NodeStatus.IDLE, data: { model: 'Wan2.2_I2V_A14B_distilled', aspectRatio: "16:9" } }
        ]
      },
      {
        id: 'preset-s2v',
        name: '文生数字人视频工作流',
        updatedAt: Date.now(),
        isDirty: false,
        isRunning: false,
        env: {
          lightx2v_url: "",
          lightx2v_token: ""
        },
        globalInputs: {},
        history: [],
        showIntermediateResults: false,
        connections: [
          { id: 'c1', sourceNodeId: 'node-prompt', sourcePortId: 'out-text', targetNodeId: 'node-chat', targetPortId: 'in-text' },
          { id: 'c2', sourceNodeId: 'node-chat', sourcePortId: 'image_prompt', targetNodeId: 'node-image', targetPortId: 'in-text' },
          { id: 'c3', sourceNodeId: 'node-chat', sourcePortId: 'speech_text', targetNodeId: 'node-tts', targetPortId: 'in-text' },
          { id: 'c4', sourceNodeId: 'node-chat', sourcePortId: 'tone', targetNodeId: 'node-tts', targetPortId: 'in-context-tone' },
          { id: 'c5', sourceNodeId: 'node-image', sourcePortId: 'out-image', targetNodeId: 'node-avatar', targetPortId: 'in-image' },
          { id: 'c6', sourceNodeId: 'node-tts', sourcePortId: 'out-audio', targetNodeId: 'node-avatar', targetPortId: 'in-audio' },
          { id: 'c7', sourceNodeId: 'node-chat', sourcePortId: 'avatar_video_prompt', targetNodeId: 'node-avatar', targetPortId: 'in-text' }
        ],
        nodes: [
          { id: 'node-prompt', toolId: 'text-input', x: 50, y: 200, status: NodeStatus.IDLE, data: { value: "一只哈士奇程序员狗，戴着耳机和工卡在办公，吐槽自己的程序员日常。用一些网络热梗。" } },
          { id: 'node-chat', toolId: 'text-generation', x: 450, y: 200, status: NodeStatus.IDLE, data: { 
              model: 'deepseek-v3-2-251201',
              mode: 'custom',
              customInstruction: `你是一位专业的数字人视频脚本编写者。你的任务是根据用户的输入描述，为数字人视频创建完整的脚本包。
    
    重要提示：
    - 生成结果中的所有字段语言必须跟随用户输入的语言。如果用户使用中文输入，所有输出字段（speech_text、tone、image_prompt、avatar_video_prompt）都必须使用中文；如果用户使用英文输入，则所有输出字段都使用英文。
    - 生成高质量、自然且引人入胜的数字人视频内容。
    
    对于 speech_text（语音文本）：
    - 编写自然、对话式的脚本，听起来真实可信
    - 保持简洁（正常语速下约20-40秒）
    - 使用清晰、直接的语言，符合角色的个性和风格
    - 确保脚本流畅自然，易于理解
    - 重要：不要在语音文本中使用括号、方括号或任何标记来表示语气或情感
    - 所有语气、情感和声音指令都应放在 'tone' 字段中，而不是 speech_text 中
    - 编写纯对话文本，不包含任何舞台指示或语气标记
    
    对于 tone（语调指令）：
    - 提供详细的配音指导，捕捉角色的个性
    - 包含情感提示、节奏、重音点和声音特征
    - 描述声音应该听起来如何（例如：温暖、权威、友好、严肃）
    - 包含语气应该转换或强调某些词语/短语的具体时刻
    - 使其对TTS系统具有可操作性，以产生自然的声音
    
    对于 image_prompt（肖像提示）：
    - 创建与描述角色匹配的详细肖像描述
    - 包含面部特征、年龄、表情、服装、背景、光线
    - 确保描述适合肖像图像生成
    - 匹配用户输入中角色的个性和风格
    - 包含将使头像看起来专业且引人入胜的视觉细节
    
    对于 avatar_video_prompt（数字人视频动作提示）：
    - 描述自然、真实的说话手势和动作
    - 包含头部动作（点头、倾斜、轻微转动）
    - 描述与语音内容和情感语调匹配的面部表情
    - 如果合适，包含肢体语言和手势
    - 指定眼神接触和视线方向
    - 确保动作与语音节奏同步
    - 足够详细，以指导数字人视频生成自然、逼真的效果
    
    以JSON格式输出所有四个字段。`,
              customOutputs: [
                { id: 'speech_text', label: '语音脚本', description: '人物对听众说的话。' },
                { id: 'tone', label: '语调指令', description: '语音风格的提示。' },
                { id: 'image_prompt', label: '肖像提示', description: '用于图像生成器的人物的肖像描述。' },
                { id: 'avatar_video_prompt', label: '数字人视频动作提示', description: '数字人视频动作和运动的描述（例如：自然的说话手势、头部动作、面部表情）。' }
              ]
          } },
          { id: 'node-image', toolId: 'text-to-image', x: 900, y: 50, status: NodeStatus.IDLE, data: { model: 'Qwen-Image-2512', aspectRatio: "9:16" } },
          { id: 'node-tts', toolId: 'tts', x: 900, y: 350, status: NodeStatus.IDLE, data: { model: 'lightx2v', voiceType: 'zh_male_dayi_saturn_bigtts', resourceId: 'seed-tts-2.0' } },
          { id: 'node-avatar', toolId: 'avatar-gen', x: 1500, y: 200, status: NodeStatus.IDLE, data: {} }
        ]
      },
];

/**
 * 获取预设工作流列表，并在运行时更新 token
 * 使用统一的 getAccessToken 函数，它会优先使用初始化时设置的全局 token
 */
export function getPresetWorkflows(): WorkflowState[] {
  // 使用统一的 getAccessToken 函数获取 token（已考虑用户登录状态）
  const token = getAccessToken();
  
  // 返回更新了 token 的预设工作流副本
  return PRESET_WORKFLOWS.map(workflow => ({
    ...workflow,
    env: {
      ...workflow.env,
      lightx2v_token: token
    }
  }));
}