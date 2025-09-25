<script setup>
import { alert, getAlertClass, getAlertIconBgClass, getAlertIcon } from '../utils/other'
import { useI18n } from 'vue-i18n'
const { t, locale } = useI18n()
</script>
<template>
            <!-- 增强的提示消息系统 -->
            <div v-cloak>
                <div v-if="alert.show"
                    class="fixed top-4 left-1/2 transform -translate-x-1/2 z-[9999] max-w-xs w-full px-4"
                    :class="getAlertClass(alert.type)">
                        <div
                            class="bg-gray-800/95 backdrop-blur-sm border border-gray-700/50 rounded-lg shadow-lg transition-all duration-300 ease-out">
                        <div class="flex items-center p-3">
                            <div class="flex-shrink-0 mr-3">
                                <div class="w-6 h-6 rounded-full flex items-center justify-center"
                                    :class="getAlertIconBgClass(alert.type)">
                                    <i :class="getAlertIcon(alert.type)" class="text-xs"></i>
                                </div>
                            </div>
                            <div class="flex-1">
                                <p class="text-xs font-medium text-gray-200">
                                    {{ alert.message }}
                                </p>
                            </div>
                            <div class="flex-shrink-0 ml-2">
                                <button @click="alert.show = false"
                                        class="w-5 h-5 rounded-full flex items-center justify-center text-gray-400 hover:text-gray-200 hover:bg-gray-700/50 transition-all duration-200">
                                    <i class="fas fa-times text-xs"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
</template>