<script setup>
import {  switchToCreateView, switchToProjectsView, switchToInspirationView } from '../utils/other'
import { useI18n } from 'vue-i18n'
const { t, locale } = useI18n()
</script>
<template>
<div class="w-full min-h-screen flex items-center justify-center p-2 sm:p-4 md:p-6">
  <div class="wrong-content">
    <div class="wrong-number">404</div>
    <h1 class="wrong-title">Page not found</h1>
    <p class="wrong-description">Sorry, we couldn't find the page you're looking for.</p>
    <div>
      <RouterLink to="/generate" @click="switchToCreateView" class="btn-primary">Go back home</RouterLink>
    </div>
  </div>
</div>
</template>

<style scoped>

.wrong-content {
  text-align: center;
  max-width: 600px;
  width: 100%;
  margin: 0 auto;
}

.wrong-number {
  font-size: 18px;
  font-weight: 600;
  color: #6A3FFB;
  margin-bottom: 16px;
}

.wrong-title {
  font-size: 64px;
  font-weight: 600;
  color: #1A1A1A;
  margin: 0 0 24px 0;
  line-height: 1.1;
}

.wrong-description {
  font-size: 20px;
  font-weight: 500;
  color: #5C5C5C;
  margin: 0 0 40px 0;
  line-height: 1.4;
}

.btn-primary {
  background-color: #6A3FFB;
  color: white;
  padding: 12px 24px;
  border-radius: 8px;
  text-decoration: none;
  font-size: 14px;
  font-weight: 600;
  border: none;
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.btn-primary:hover {
  background-color: #5A2FE8;
}

.btn-secondary {
  color: #1A1A1A;
  text-decoration: none;
  font-size: 14px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 4px;
}

.btn-secondary:hover {
  text-decoration: underline;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .wrong-page {
    padding: 16px;
  }
  
  .wrong-title {
    font-size: 48px;
  }
  
  .wrong-description {
    font-size: 18px;
  }

  .btn-primary,
  .btn-secondary {
    width: 100%;
    justify-content: center;
  }
}
</style>
