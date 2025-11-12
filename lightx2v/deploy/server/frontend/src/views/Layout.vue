<script setup>
import { ref } from 'vue'
import TopBar from '../components/TopBar.vue'
import LeftBar from '../components/LeftBar.vue'
import Alert from '../components/Alert.vue'
import Confirm from '../components/Confirm.vue'
import TaskDetails from '../components/TaskDetails.vue'
import TemplateDetails from '../components/TemplateDetails.vue'
import PromptTemplate from '../components/PromptTemplate.vue'
import Voice_tts from '../components/Voice_tts.vue'
import VoiceTtsHistoryPanel from '../components/VoiceTtsHistoryPanel.vue'
import MediaTemplate from '../components/MediaTemplate.vue'
import Loading from '../components/Loading.vue'
import SiteFooter from '../components/SiteFooter.vue'
import { useI18n } from 'vue-i18n'
import { isLoading, showVoiceTTSModal, handleAudioUpload, showAlert, ttsHistory, loadTtsHistory } from '../utils/other'

const { t } = useI18n()

const voiceTtsRef = ref(null)
const showCombinedHistory = ref(false)
const showTextHistory = ref(false)
const showInstructionHistory = ref(false)
const showVoiceHistory = ref(false)

const closeAllHistoryPanels = () => {
    showCombinedHistory.value = false
    showTextHistory.value = false
    showInstructionHistory.value = false
    showVoiceHistory.value = false
}

const handleOpenHistoryPanel = async (mode) => {
    await loadTtsHistory()
    closeAllHistoryPanels()
    if (mode === 'combined') showCombinedHistory.value = true
    else if (mode === 'text') showTextHistory.value = true
    else if (mode === 'instruction') showInstructionHistory.value = true
    else if (mode === 'voice') showVoiceHistory.value = true
}

const applyCombinedHistory = async (entry) => {
    await voiceTtsRef.value?.applyCombinedHistoryEntry(entry)
    closeAllHistoryPanels()
}

const applyTextHistory = (value) => {
    voiceTtsRef.value?.applyTextHistoryEntry(value)
    closeAllHistoryPanels()
}

const applyInstructionHistory = (value) => {
    voiceTtsRef.value?.applyInstructionHistoryEntry(value)
    closeAllHistoryPanels()
}

const applyVoiceHistory = async (value) => {
    await voiceTtsRef.value?.applyVoiceHistoryEntry(value)
    closeAllHistoryPanels()
}

const deleteHistoryEntry = (entry) => {
    voiceTtsRef.value?.handleDeleteHistoryEntry(entry)
}

const getHistoryVoiceName = (entry) => voiceTtsRef.value?.getHistoryVoiceName(entry) || ''

// 处理 TTS 完成回调
const handleTTSComplete = (audioBlob) => {
    // 创建File对象
    const audioFile = new File([audioBlob], 'tts_audio.mp3', { type: 'audio/mpeg' })

    // 模拟文件上传事件
    const dataTransfer = new DataTransfer()
    dataTransfer.items.add(audioFile)
    const fileList = dataTransfer.files

    const event = {
        target: {
            files: fileList
        }
    }

    // 处理音频上传
    handleAudioUpload(event)

    // 关闭模态框
    showVoiceTTSModal.value = false

    // 显示成功提示
    showAlert('语音合成完成，已自动添加到音频素材', 'success')
}
</script>

<template>
  <!-- 主容器 - Apple 极简风格 - 配合80%缩放铺满屏幕 -->
  <div class="bg-[#f5f5f7] dark:bg-[#000000] transition-colors duration-300 w-full h-full">
    <!-- 主内容区域 -->
    <div class="flex flex-col w-full h-full">
      <!-- 顶部导航栏 -->
      <TopBar />

      <!-- 内容区域 - 响应式布局 -->
      <div class="flex flex-col sm:flex-row flex-1 h-full">
        <!-- 左侧/底部导航栏 - 响应式 -->
        <LeftBar />

        <!-- 路由视图内容 -->
        <div class="flex-1 overflow-y-auto main-scrollbar">
          <router-view></router-view>
        </div>
      </div>

      <SiteFooter />
    </div>

    <!-- 全局组件 -->
    <Alert />
    <Confirm />
    <TaskDetails />
    <TemplateDetails />
    <PromptTemplate />
    <Voice_tts
      v-if="showVoiceTTSModal"
      ref="voiceTtsRef"
      @tts-complete="handleTTSComplete"
      @close-modal="showVoiceTTSModal = false"
      @open-history-panel="handleOpenHistoryPanel"
    />
    <MediaTemplate />

    <VoiceTtsHistoryPanel
      :visible="showCombinedHistory"
      :history="ttsHistory"
      mode="combined"
      :get-voice-name="getHistoryVoiceName"
      @close="closeAllHistoryPanels"
      @apply="applyCombinedHistory"
      @delete="deleteHistoryEntry"
    />

    <VoiceTtsHistoryPanel
      :visible="showTextHistory"
      :history="ttsHistory"
      mode="text"
      @close="closeAllHistoryPanels"
      @apply="applyTextHistory"
    />

    <VoiceTtsHistoryPanel
      :visible="showInstructionHistory"
      :history="ttsHistory"
      mode="instruction"
      @close="closeAllHistoryPanels"
      @apply="applyInstructionHistory"
    />

    <VoiceTtsHistoryPanel
      :visible="showVoiceHistory"
      :history="ttsHistory"
      mode="voice"
      :get-voice-name="getHistoryVoiceName"
      @close="closeAllHistoryPanels"
      @apply="applyVoiceHistory"
    />

    <!-- 全局加载覆盖层 - Apple 风格 -->
    <div v-show="isLoading" class="fixed inset-0 bg-[#f5f5f7] dark:bg-[#000000] flex items-center justify-center z-[9999] transition-opacity.duration-300">
      <Loading />
    </div>
  </div>
</template>

<style scoped>
/* Apple 风格极简设计 - 所有样式已通过 Tailwind CSS 在 template 中定义 */
</style>
