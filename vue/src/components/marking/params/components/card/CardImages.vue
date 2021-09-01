<template>
  <div class="t-card-images">
    <div class="t-card-images__left"></div>
    <div class="t-card-images__center"></div>
    <div class="t-card-images__right"></div>
  </div>
</template>

<script>
export default {
  name: 't-card-images',
  props: {
    label: String,
    type: String,
  },
  computed: {
  },
};
</script>

<style lang="scss" scoped>
.t-card-images {
  
}
</style>