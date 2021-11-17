<template>
  <div v-if="info" class="info">
    <div class="info__title">Информация о блоке</div>
    <div class="info__content">
      <p v-html="info"></p>
    </div>
  </div>
</template>

<script>
export default {
  name: 'info-block',
  props: {
    info: String,
  },
};
</script>

<style lang="scss" scoped>
.info {
  &__title {
    background-color: #0e1621;
    margin: 10px -20px;
    padding: 4px 20px;
    font-size: 0.75rem;
    color: #ffffff;
  }
  &__content {
    p {
      font-size: 14px;
      color: #a7bed3;
    }
  }
}
</style>