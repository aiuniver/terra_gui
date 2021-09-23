<template>
  <div class="t-predict-image">
    <img width="auto" height="120" :src="src" :alt="'value'" />
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
  },
  computed: {
    src() {
      return `/_media/blank/?hash=${this.value}`
      // return window.location.host
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
