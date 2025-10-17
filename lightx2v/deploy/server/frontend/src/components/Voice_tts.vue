<template>
     <!-- 模态框头部 -->
     <div class="flex items-center justify-between p-4 border-b border-gray-700">
     <h3 class="text-xl font-semibold text-white">语音合成</h3>
     <div class="flex items-center gap-2">
       <button
         @click="applySelectedVoice"
         :disabled="!selectedVoice || !inputText.trim() || isGenerating"
         class="header-apply-button"
         title="应用当前选择的声音"
       >
         <i class="fas fa-check"></i>
       </button>
       <button @click="closeModal"
           class="w-8 h-8 flex items-center justify-center text-gray-400 hover:text-white hover:bg-gray-700 rounded-full transition-all duration-200">
           <i class="fas fa-times text-lg"></i>
       </button>
     </div>
     </div>
    <!-- 模态框内容 -->
<div class="h-full overflow-y-auto p-4 main-scrollbar">
  <div class="voice-tts-container">

    <!-- Text Input Section -->
    <div class="text-input-section">
      <h6 class="section-title">输入要转换的文本</h6>
      <div class="input-wrapper">
        <textarea
          v-model="inputText"
          placeholder="你好，请问我有什么可以帮您？"
          class="text-input"
          rows="4"
        ></textarea>
      </div>
    </div>

        <!-- Text Input Section -->
     <div class="text-input-section">
      <h6 class="section-title">语音指令（仅适用于v2.0音色）</h6>
      <div class="input-wrapper">
        <textarea
          v-model="contextText"
          placeholder="使用指令控制合成语音细节，包括但不限于情绪、语境、方言、语气、速度、音调等（仅适用于v2.0音色），例如：带点害羞又藏着温柔期待的语气说"
          class="text-input"
          rows="1"
        ></textarea>
      </div>
    </div>

    <!-- Voice Selection Section -->
    <div class="voice-selection-section">
      <div class="header-controls">
        <h6 class="section-title">选择音色</h6>
        <div class="filter-controls">
          <div class="search-wrapper">
            <div class="arco-input-group-wrapper arco-input-group-wrapper-default">
              <span class="arco-input-group">
                <span class="arco-input-inner-wrapper arco-input-inner-wrapper-has-prefix arco-input-inner-wrapper-default">
                  <span class="arco-input-group-prefix">
                    <i class="fas fa-search"></i>
                  </span>
                  <input
                    v-model="searchQuery"
                    placeholder="搜索音色"
                    class="arco-input arco-input-size-default"
                  />
                </span>
              </span>
            </div>
          </div>

          <!-- Filter Button -->
          <button @click="toggleFilterPanel" class="filter-button">
            <i class="fas fa-filter"></i>
            <span>筛选</span>
          </button>
        </div>
      </div>


      <!-- Voice List -->
      <div class="voice-list-container" ref="voiceListContainer">
        <div class="arco-radio-group arco-radio-size-default arco-radio-mode-outline" role="radiogroup">
          <div class="voice-grid">
            <label
              v-for="(voice, index) in filteredVoices"
              :key="index"
              class="arco-radio m-0 p-0"
              :class="{ 'arco-radio-checked': selectedVoice === voice.voice_type }"
            >
              <input
                type="radio"
                :value="voice.voice_type"
                v-model="selectedVoice"
                @change="onVoiceSelect(voice)"
              />
              <div
                class="voice-card"
                :class="{
                  'voice-card-selected': selectedVoice === voice.voice_type,
                  'voice-card-controlled': selectedVoice === voice.voice_type && showControls
                }"
              >
                <!-- V2 Tag -->
                <div v-if="voice.version === '2.0'" class="version-tag">
                  v2
                </div>
                <div class="voice-avatar-container">
                  <!-- Female Avatar -->
                  <img
                    v-if="isFemaleVoice(voice.voice_type)"
                    src="../../public/female.svg"
                    alt="Female Avatar"
                    class="voice-avatar"
                    :class="{ 'voice-avatar-disabled': selectedVoice === voice.voice_type }"
                  />
                  <!-- Male Avatar -->
                  <img
                    v-else
                    src="../../public/male.svg"
                    alt="Male Avatar"
                    class="voice-avatar"
                    :class="{ 'voice-avatar-disabled': selectedVoice === voice.voice_type }"
                  />
                  <!-- Loading indicator -->
                  <div v-if="isGenerating && selectedVoice === voice.voice_type" class="loading-indicator">
                    <i class="fas fa-spinner fa-spin"></i>
                  </div>
                  <!-- Settings button for selected voice -->
                  <div v-if="!isGenerating && selectedVoice === voice.voice_type" class="settings-button" @click="toggleControls">
                    <i class="fas fa-cog"></i>
                  </div>
                </div>
                <div class="voice-info">
                  <div class="voice-name">
                    {{ voice.name }}
                  </div>
                  <div class="voice-tags">
                    <div class="voice-scene" v-if="voice.scene">
                      <span class="scene-tag">{{ voice.scene }}</span>
                    </div>
                    <div class="voice-languages" v-if="voice.language && voice.language.length > 0">
                      <span
                        v-for="langCode in voice.language"
                        :key="langCode"
                        class="language-tag"
                      >
                        {{ getLanguageDisplayName(langCode) }}
                      </span>
                    </div>
                  </div>
                </div>

                <!-- TTS Controls Panel -->
                <div v-if="selectedVoice === voice.voice_type && showControls" class="voice-controls-panel">
                  <div class="control-item">
                    <label>语速:</label>
                    <input
                      type="range"
                      min="-50"
                      max="100"
                      v-model="speedRate"
                      class="mini-slider"
                    />
                    <span class="mini-display">{{ getSpeedDisplayValue(speedRate) }}</span>
                  </div>
                  <div class="control-item">
                    <label>音量:</label>
                    <input
                      type="range"
                      min="-100"
                      max="100"
                      v-model="loudnessRate"
                      class="mini-slider"
                    />
                    <span class="mini-display">{{ getLoudnessDisplayValue(loudnessRate) }}</span>
                  </div>
                  <!-- Emotion controls - only show if voice has emotions -->
                  <div v-if="voice.emotions && voice.emotions.length > 0" class="control-item">
                    <label>情感强度:</label>
                    <input
                      type="range"
                      min="1"
                      max="5"
                      v-model="emotionScale"
                      class="mini-slider"
                    />
                    <span class="mini-display">{{ emotionScale }}</span>
                  </div>
                  <div v-if="voice.emotions && voice.emotions.length > 0" class="control-item">
                    <label>情感类型:</label>
                    <DropdownMenu
                      :items="emotionItems"
                      :selected-value="selectedEmotion"
                      placeholder="中性"
                      @select-item="handleEmotionSelect"
                      class="mini-dropdown"
                    />
                  </div>
                </div>
              </div>
            </label>
          </div>
        </div>
      </div>
    </div>

    <!-- Audio Player -->
    <div class="audio-player" v-if="audioUrl">
      <h6>生成的音频:</h6>
      <audio :src="audioUrl" controls class="audio-controls"></audio>
    </div>
  </div>

  <!-- Filter Panel Overlay -->
  <div v-if="showFilterPanel" class="filter-overlay" @click="closeFilterPanel">
    <div class="filter-panel" @click.stop>
      <!-- Filter Panel Header -->
      <div class="filter-header">
        <h3 class="filter-title">筛选</h3>
        <button @click="closeFilterPanel" class="filter-close-btn">
          <i class="fas fa-times"></i>
        </button>
      </div>

      <!-- Filter Content -->
      <div class="filter-content">
        <!-- Scene Filter -->
        <div class="filter-section">
          <h4 class="filter-section-title">场景</h4>
          <div class="filter-options">
            <button
              v-for="category in categories"
              :key="category"
              @click="selectCategory(category)"
              class="filter-option"
              :class="{ 'filter-option-selected': selectedCategory === category }"
            >
              {{ category }}
            </button>
          </div>
        </div>

        <!-- Version Filter -->
        <div class="filter-section">
          <h4 class="filter-section-title">版本</h4>
          <div class="filter-options">
            <button
              v-for="v in version"
              :key="v"
              @click="selectVersion(v)"
              class="filter-option"
              :class="{ 'filter-option-selected': selectedVersion === v }"
            >
              {{ v }}
            </button>
          </div>
        </div>

        <!-- Language Filter -->
        <div class="filter-section">
          <h4 class="filter-section-title">语言</h4>
          <div class="filter-options">
            <button
              v-for="lang in languages"
              :key="lang"
              @click="selectLanguage(lang)"
              class="filter-option"
              :class="{ 'filter-option-selected': selectedLanguage === lang }"
            >
              {{ lang }}
            </button>
          </div>
        </div>

        <!-- Gender Filter -->
        <div class="filter-section">
          <h4 class="filter-section-title">性别</h4>
          <div class="filter-options">
            <button
              v-for="gender in genders"
              :key="gender"
              @click="selectGender(gender)"
              class="filter-option"
              :class="{ 'filter-option-selected': selectedGender === gender }"
            >
              {{ gender }}
            </button>
          </div>
        </div>
      </div>

      <!-- Filter Actions -->
      <div class="filter-actions">
        <button @click="resetFilters" class="filter-reset-btn">重置</button>
        <button @click="applyFilters" class="filter-apply-btn">完成</button>
      </div>
    </div>
  </div>
</div>
</template>

<script>
import { ref, computed, onMounted, watch } from 'vue'
import DropdownMenu from './DropdownMenu.vue'

export default {
  name: 'VoiceTTS',
  components: {
    DropdownMenu
  },
  emits: ['tts-complete', 'close-modal'],
  setup(props, { emit }) {
    const inputText = ref('')
    const contextText = ref('')
    const selectedVoice = ref('')
    const selectedVoiceResourceId = ref('')
    const searchQuery = ref('')
    const speedRate = ref(0)
    const loudnessRate = ref(0)
    const emotionScale = ref(3)
    const selectedEmotion = ref('neutral')
    const isGenerating = ref(false)
    const audioUrl = ref('')
    const voices = ref([])
    const emotions = ref([])
    const voiceListContainer = ref(null)
    const showControls = ref(false)
    const showFilterPanel = ref(false)

    // Category filtering
    const selectedCategory = ref('全部场景')
    const categories = ref(['全部场景', '通用场景', '客服场景', '教育场景', '趣味口音', '角色扮演', '有声阅读', '多语种', '多情感', '视频配音'])
    const selectedVersion = ref('全部版本')
    const version = ref(['全部版本', '1.0', '2.0'])
    const selectedLanguage = ref('全部语言')
    const languages = ref(['全部语言'])
    const selectedGender = ref('全部性别')
    const genders = ref(['全部性别'])


    // Load voices data
    onMounted(async () => {
      try {
        const response = await fetch('/api/v1/voices/list')
        const data = await response.json()
        console.log('音色数据', data)
        voices.value = data.voices || []
        emotions.value = data.emotions || []

        // Map languages data to language options
        if (data.languages && Array.isArray(data.languages)) {
          const languageOptions = ['全部语言']
          data.languages.forEach(lang => {
            languageOptions.push(lang.zh) // Use Chinese name
          })
          languages.value = languageOptions
        }

        // Extract gender options from voices data
        if (voices.value && voices.value.length > 0) {
          const genderSet = new Set()
          voices.value.forEach(voice => {
            if (voice.gender) {
              genderSet.add(voice.gender)
            }
          })

          const genderOptions = ['全部性别']
          // Convert English gender to Chinese display
          genderSet.forEach(gender => {
            if (gender === 'female') {
              genderOptions.push('女性')
            } else if (gender === 'male') {
              genderOptions.push('男性')
            } else {
              // For any other gender values, use as is
              genderOptions.push(gender)
            }
          })
          genders.value = genderOptions
        }
      } catch (error) {
        console.error('Failed to load voices:', error)
      }
    })

    // 监听参数变化，自动重新生成音频
    watch([speedRate, loudnessRate, emotionScale, selectedEmotion], () => {
      if (selectedVoice.value && inputText.value.trim() && !isGenerating.value) {
        generateTTS()
      }
    })

    // 监听文本输入变化，使用防抖避免频繁生成
    let textTimeout = null
    watch([inputText, contextText], () => {
      if (textTimeout) {
        clearTimeout(textTimeout)
      }
      textTimeout = setTimeout(() => {
        if (selectedVoice.value && inputText.value.trim() && !isGenerating.value) {
          generateTTS()
        }
      }, 800) // 延迟800ms执行，给用户足够时间输入
    })

    // 监听搜索查询变化，重置滚动位置（延迟执行以避免频繁重置）
    let searchTimeout = null
    watch(searchQuery, () => {
      if (searchTimeout) {
        clearTimeout(searchTimeout)
      }
      searchTimeout = setTimeout(() => {
        resetScrollPosition()
      }, 300) // 延迟300ms执行
    })

    // Filter voices based on search query, category, version, language, and gender
    // Emotion items for dropdown
    const emotionItems = computed(() => {
      const items = []
      const selectedVoiceData = voices.value.find(v => v.voice_type === selectedVoice.value)
      if (selectedVoiceData && selectedVoiceData.emotions && emotions.value.length > 0) {
        selectedVoiceData.emotions.forEach(emotionName => {
          // Find the emotion data from emotions array
          const emotionData = emotions.value.find(emotion => emotion.name === emotionName)
          if (emotionData) {
            items.push({ value: emotionName, label: emotionData.zh })
          } else {
            // Fallback if emotion not found in emotions data
            items.push({ value: emotionName, label: emotionName })
          }
        })
      }

      // If no emotions found or no neutral emotion in the list, add neutral as default
      if (items.length === 0 || !items.find(item => item.value === 'neutral')) {
        items.unshift({ value: 'neutral', label: '中性' })
      }

      return items
    })

    const filteredVoices = computed(() => {
      let filtered = [...voices.value] // 创建副本，避免修改原始数据

      console.log('原始音色数据:', voices.value.length)
      console.log('筛选条件:', {
        category: selectedCategory.value,
        version: selectedVersion.value,
        language: selectedLanguage.value,
        gender: selectedGender.value,
        search: searchQuery.value
      })

      // Filter by category
      if (selectedCategory.value !== '全部场景') {
        filtered = filtered.filter(voice => voice.scene === selectedCategory.value)
        console.log('分类筛选后:', filtered.length)
      }

      // Filter by version
      if (selectedVersion.value !== '全部版本') {
        filtered = filtered.filter(voice => voice.version === selectedVersion.value)
        console.log('版本筛选后:', filtered.length)
      }

      // Filter by language
      if (selectedLanguage.value !== '全部语言') {
        // Convert Chinese language display back to language code for filtering
        let languageFilter = selectedLanguage.value
        // Create a mapping from Chinese names to language codes
        const languageMap = {
          '中文': 'chinese',
          '美式英语': 'en_us',
          '英式英语': 'en_gb',
          '澳洲英语': 'en_au',
          '西语': 'es',
          '日语': 'ja'
        }

        if (languageMap[selectedLanguage.value]) {
          languageFilter = languageMap[selectedLanguage.value]
        }

        filtered = filtered.filter(voice => {
          // Check if voice.language array contains the language code
          return voice.language && Array.isArray(voice.language) && voice.language.includes(languageFilter)
        })
        console.log('语言筛选后:', filtered.length)
      }

      // Filter by gender
      if (selectedGender.value !== '全部性别') {
        // Convert Chinese gender display back to English for filtering
        let genderFilter = selectedGender.value
        if (selectedGender.value === '女性') {
          genderFilter = 'female'
        } else if (selectedGender.value === '男性') {
          genderFilter = 'male'
        }

        filtered = filtered.filter(voice => voice.gender === genderFilter)
        console.log('性别筛选后:', filtered.length)
      }

      // Filter by search query
      if (searchQuery.value) {
        filtered = filtered.filter(voice =>
          voice.name.toLowerCase().includes(searchQuery.value.toLowerCase())
        )
        console.log('搜索筛选后:', filtered.length)
      }

      console.log('最终筛选结果:', filtered.length)
      return filtered
    })

    // Check if voice is female based on name
    const isFemaleVoice = (name) => {
      return name.toLowerCase().includes('female')
    }

    // Get available emotions for selected voice
    const availableEmotions = computed(() => {
      const selectedVoiceData = voices.value.find(v => v.voice_type === selectedVoice.value)
      return selectedVoiceData?.emotions || []
    })

    // Reset scroll position
    // Handle emotion selection
    const handleEmotionSelect = (item) => {
      selectedEmotion.value = item.value
    }

    const resetScrollPosition = () => {
      // Reset voice list scroll to top
      if (voiceListContainer.value) {
        voiceListContainer.value.scrollTop = 0
      }
    }

    // Handle voice selection and auto-generate TTS
    const onVoiceSelect = async (voice) => {
      selectedVoice.value = voice.voice_type
      selectedVoiceResourceId.value = voice.resource_id
      // Reset emotion if not available for this voice
      if (voice.emotions && !voice.emotions.includes(selectedEmotion.value)) {
        selectedEmotion.value = ''
      }

      // Reset scroll position when voice is selected
      resetScrollPosition()

      // Auto-generate TTS when voice is selected and text is available
      await generateTTS()
    }

    // Generate TTS and auto-play
    const generateTTS = async () => {
      if (!inputText.value.trim()) {
        inputText.value = '你好，请问我有什么可以帮您？'
      }

      if (!selectedVoice.value) return
      console.log('contextText', contextText.value)
      isGenerating.value = true
      try {
        const response = await fetch('/api/v1/tts/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: inputText.value,
            voice_type: selectedVoice.value,
            context_texts: contextText.value,
            emotion: selectedEmotion.value,
            emotion_scale: emotionScale.value,
            speed_rate: speedRate.value,
            loudness_rate: loudnessRate.value,
            resource_id: selectedVoiceResourceId.value
          })
        })

        if (response.ok) {
          const blob = await response.blob()
          audioUrl.value = URL.createObjectURL(blob)

          // 自动播放生成的音频
          const audio = new Audio(audioUrl.value)
          audio.play().catch(error => {
            console.log('自动播放被阻止:', error)
            // 如果自动播放失败，用户仍可以手动播放
          })
        } else {
          throw new Error('TTS generation failed')
        }
      } catch (error) {
        console.error('TTS generation error:', error)
        alert('语音生成失败，请重试')
      } finally {
        isGenerating.value = false
      }
    }

    // Apply selected voice (emit the generated audio)
    const applySelectedVoice = () => {
      if (audioUrl.value) {
        // Convert the audio URL back to blob and emit
        fetch(audioUrl.value)
          .then(response => response.blob())
          .then(blob => {
            emit('tts-complete', blob)
          })
          .catch(error => {
            console.error('Error converting audio to blob:', error)
            alert('应用音频失败，请重试')
          })
      }
    }

    // Close modal function
    const closeModal = () => {
      emit('close-modal')
    }

    // Toggle controls panel
    const toggleControls = () => {
      showControls.value = !showControls.value
    }

    // Filter panel functions
    const toggleFilterPanel = () => {
      showFilterPanel.value = !showFilterPanel.value
    }

    const closeFilterPanel = () => {
      showFilterPanel.value = false
    }

    const selectCategory = (category) => {
      selectedCategory.value = category
    }

    const selectVersion = (version) => {
      selectedVersion.value = version
    }

    const selectLanguage = (language) => {
      selectedLanguage.value = language
    }

    const selectGender = (gender) => {
      selectedGender.value = gender
    }

    const resetFilters = () => {
      selectedCategory.value = '全部场景'
      selectedVersion.value = '全部版本'
      selectedLanguage.value = '全部语言'
      selectedGender.value = '全部性别'
    }

    const applyFilters = () => {
      showFilterPanel.value = false
      resetScrollPosition()
    }

    // Convert speed rate to display value (0.5x to 2.0x)
    const getSpeedDisplayValue = (value) => {
      // Map -50 to 100 range to 0.5x to 2.0x
      const ratio = (parseInt(value) + 50) / 150 // Convert to 0-1 range
      const speed = 0.5 + (ratio * 1.5) // Convert to 0.5-2.0 range
      return `${speed.toFixed(1)}x`
    }

    // Convert loudness rate to display value (-100 to 100)
    const getLoudnessDisplayValue = (value) => {
      // Map -100 to 100 range to -50 to 100 for API
      const apiValue = Math.round((parseInt(value) + 100) * 150 / 200 - 50)
      return `${value}%`
    }

    // Convert language code to Chinese display name
    const getLanguageDisplayName = (langCode) => {
      const languageMap = {
        'chinese': '中文',
        'en_us': '美式英语',
        'en_gb': '英式英语',
        'en_au': '澳洲英语',
        'es': '西语',
        'ja': '日语'
      }
      return languageMap[langCode] || langCode
    }


    return {
      inputText,
      contextText,
      selectedVoice,
      searchQuery,
      speedRate,
      loudnessRate,
      emotionScale,
      selectedEmotion,
      isGenerating,
      audioUrl,
      voices,
      voiceListContainer,
      showControls,
      showFilterPanel,
      filteredVoices,
      isFemaleVoice,
      availableEmotions,
      onVoiceSelect,
      generateTTS,
      applySelectedVoice,
      closeModal,
      toggleControls,
      toggleFilterPanel,
      closeFilterPanel,
      selectCategory,
      selectVersion,
      selectLanguage,
      selectGender,
      resetFilters,
      applyFilters,
      getSpeedDisplayValue,
      getLoudnessDisplayValue,
      getLanguageDisplayName,
      emotionItems,
      handleEmotionSelect,
      selectedCategory,
      categories,
      selectedVoiceResourceId,
      version,
      selectedVersion,
      selectedLanguage,
      languages,
      selectedGender,
      genders,
      resetScrollPosition
    }
  }
}
</script>

<style scoped>
.voice-tts-container {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
  position: relative;
  overflow: visible;
  height: auto;
  min-height: fit-content;
}


.header-apply-button {
  width: 32px;
  height: 32px;
  background: #8b5cf6;
  color: white;
  border: none;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 14px;
}

.header-apply-button:hover:not(:disabled) {
  background: #7c3aed;
  transform: scale(1.1);
}

.header-apply-button:disabled {
  background: #c9cdd4;
  cursor: not-allowed;
  transform: none;
}

.text-input-section {
  margin-bottom: 30px;
}

.section-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 12px;
  color: #ffffff;
}

.input-wrapper {
  margin-bottom: 16px;
}

.text-input {
  width: 100%;
  padding: 12px;
  border: 1px solid #374151;
  border-radius: 6px;
  font-size: 14px;
  resize: vertical;
  min-height: 100px;
  background: #374151;
  color: #ffffff;
}

.text-input:focus {
  outline: none;
  border-color: #8b5cf6;
  box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.1);
}

.voice-selection-section {
  margin-bottom: 30px;
  position: relative;
  overflow: visible;
  height: auto;
}

.header-controls {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 16px;
  flex-wrap: wrap;
  gap: 16px;
}

.filter-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.search-wrapper {
  width: 218px;
  flex-shrink: 0;
}

.filter-button {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 6px;
  color: #ffffff;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.filter-button:hover {
  background: rgba(255, 255, 255, 0.2);
  border-color: rgba(255, 255, 255, 0.5);
}

.arco-input-group-wrapper {
  width: 100%;
}

.arco-input-group {
  display: flex;
  align-items: center;
}

.arco-input-inner-wrapper {
  position: relative;
  display: flex;
  align-items: center;
  width: 100%;
  border: 1px solid #374151;
  border-radius: 6px;
  background: #374151;
}

.arco-input-inner-wrapper:focus-within {
  border-color: #8b5cf6;
  box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.1);
}

.arco-input-group-prefix {
  padding: 0 12px;
  display: flex;
  align-items: center;
  color: #86909c;
}

.arco-input-group-prefix svg {
  color: #86909c;
}

.arco-input {
  flex: 1;
  padding: 8px 12px;
  border: none;
  outline: none;
  font-size: 14px;
  color: #ffffff;
  background: transparent;
}

.arco-icon {
    color: #86909c;
}

.voice-list-container {
  background: #1f2937;
  border-radius: 8px;
  padding: 20px;
  max-height: 600px;
  overflow-y: auto;
  overflow-x: hidden;
  scroll-behavior: smooth;
  height: auto;
  min-height: fit-content;
}

.voice-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}

.voice-card {
  display: flex;
  align-items: center;
  padding: 16px;
  border: 1px solid #374151;
  border-radius: 8px;
  background: #374151;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
}

.voice-card:hover {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.voice-card-selected {
  border-color: #8b5cf6;
  background: rgba(139, 92, 246, 0.15);
}

.voice-card-controlled {
  border-color: #8b5cf6;
  background: rgba(139, 92, 246, 0.2);
}

.voice-avatar-container {
  position: relative;
  margin-right: 12px;
}

.voice-avatar {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  object-fit: cover;
  background: white;
  transition: all 0.3s ease;
}

.voice-avatar-disabled {
  filter: grayscale(30%) brightness(0.6);
}

.loading-indicator {
    position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 30px;
  height: 30px;
  background: rgba(139, 92, 246, 0.8);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s ease;
  z-index: 20;
}

.voice-info {
  flex: 1;
}

.voice-name {
  font-size: 14px;
  font-weight: 500;
  color: #ffffff;
  margin-bottom: 4px;
}

.voice-tags {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.voice-languages {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.scene-tag {
  display: inline-block;
  padding: 2px 8px;
  background: #4b5563;
  color: #d1d5db;
  border-radius: 4px;
  font-size: 12px;
}

.language-tag {
  display: inline-block;
  padding: 2px 6px;
  background: rgba(139, 92, 246, 0.2);
  color: #a78bfa;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
}


.audio-player {
  background: rgba(255, 255, 255, 0.05);
  padding: 20px;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.audio-player h6 {
  color: #ffffff;
  margin-bottom: 8px;
}

.audio-controls {
  width: 100%;
  margin-top: 8px;
}


/* Hide radio input */
.arco-radio input[type="radio"] {
  position: absolute;
  opacity: 0;
  pointer-events: none;
}

/* Version tag styles */
.version-tag {
  position: absolute;
  top: 8px;
  right: 8px;
  background: #8b5cf6;
  color: white;
  font-size: 10px;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 8px;
  z-index: 10;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  line-height: 1;
  min-width: 20px;
  text-align: center;
}

/* Settings button styles */
.settings-button {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 30px;
  height: 30px;
  background: rgba(139, 92, 246, 0.8);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s ease;
  z-index: 20;
}

.settings-button:hover {
  background: #7c3aed;
  transform: translate(-50%, -50%) scale(1.1);
}

/* Voice controls panel styles */
.voice-controls-panel {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: #1f2937;
  border: 1px solid #374151;
  border-radius: 8px;
  padding: 12px;
  margin-top: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  z-index: 100;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.control-item {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 6px;
}

.control-item label {
  font-size: 12px;
  color: #ffffff;
  min-width: 40px;
}

.mini-slider {
  flex: 1;
  height: 4px;
  background: #4b5563;
  border-radius: 2px;
  outline: none;
}

.mini-slider::-webkit-slider-thumb {
  appearance: none;
  width: 12px;
  height: 12px;
  background: #8b5cf6;
  border-radius: 50%;
  cursor: pointer;
}

.mini-display {
  font-size: 11px;
  color: #ffffff;
  min-width: 30px;
  text-align: center;
}

.mini-select {
  flex: 1;
  padding: 4px 8px;
  border: 1px solid #4b5563;
  border-radius: 4px;
  font-size: 12px;
  background: #374151;
  color: #ffffff;
}

.mini-dropdown {
  min-width: 100px;
  font-size: 12px;
}

.mini-dropdown :deep(.inline-flex) {
  padding: 4px 8px !important;
  font-size: 12px !important;
  min-height: 28px !important;
}

/* Filter Panel Styles */
.filter-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  z-index: 1000;
  display: flex;
  align-items: flex-end;
  justify-content: center;
}

.filter-panel {
  background: #2d2d2d;
  width: 100%;
  max-width: 600px;
  max-height: 80vh;
  border-radius: 20px 20px 0 0;
  overflow: hidden;
  animation: slideUp 0.3s ease-out;
}

@keyframes slideUp {
  from {
    transform: translateY(100%);
  }
  to {
    transform: translateY(0);
  }
}

.filter-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #404040;
}

.filter-title {
  color: #ffffff;
  font-size: 18px;
  font-weight: 600;
  margin: 0;
}

.filter-close-btn {
  width: 32px;
  height: 32px;
  border: none;
  background: none;
  color: #ffffff;
  font-size: 18px;
  cursor: pointer;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.2s ease;
}

.filter-close-btn:hover {
  background: rgba(255, 255, 255, 0.1);
}

.filter-content {
  padding: 20px;
  max-height: 60vh;
  overflow-y: auto;
}

.filter-section {
  margin-bottom: 24px;
}

.filter-section:last-child {
  margin-bottom: 0;
}

.filter-section-title {
  color: #ffffff;
  font-size: 16px;
  font-weight: 500;
  margin: 0 0 12px 0;
}

.filter-options {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.filter-option {
  padding: 8px 16px;
  background: #404040;
  border: 1px solid #555555;
  border-radius: 20px;
  color: #cccccc;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
  white-space: nowrap;
}

.filter-option:hover {
  background: #505050;
  border-color: #666666;
}

.filter-option-selected {
  background: #8b5cf6;
  border-color: #8b5cf6;
  color: #ffffff;
}

.filter-option-selected:hover {
  background: #7c3aed;
  border-color: #7c3aed;
}

.filter-actions {
  display: flex;
  gap: 12px;
  padding: 20px;
  border-top: 1px solid #404040;
}

.filter-reset-btn {
  flex: 1;
  padding: 12px;
  background: #404040;
  border: 1px solid #555555;
  border-radius: 8px;
  color: #cccccc;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.filter-reset-btn:hover {
  background: #505050;
  border-color: #666666;
}

.filter-apply-btn {
  flex: 2;
  padding: 12px;
  background: #8b5cf6;
  border: 1px solid #8b5cf6;
  border-radius: 8px;
  color: #ffffff;
  font-size: 16px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.filter-apply-btn:hover {
  background: #7c3aed;
  border-color: #7c3aed;
}

</style>
