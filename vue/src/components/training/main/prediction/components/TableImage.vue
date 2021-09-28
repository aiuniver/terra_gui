<template>
  <div class="t-predict-image">
    <img width="auto" height="120" :src="src" :alt="'value'" :key="src" />
  </div>
</template>
<script>
export default {
  name: 't-table-image',
  props: {
    value: {
      type: String,
      default: '',
    },
    update: String
  },
  computed: {
    src() {
      return `/_media/blank/?path=${this.value}&r=${this.update}`
    }
  }
};
</script>

<style lang="scss" scoped>
.t-predict-image {
  height: 160px;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 0 20px;
  // &__image {}
}
</style>
