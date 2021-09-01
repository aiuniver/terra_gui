<template>
  <div class="t-card-marking">
    <div class="t-card-marking__left"></div>
    <div class="t-card-marking__center"></div>
    <div class="t-card-marking__right"></div>
  </div>
</template>

<script>
export default {
  name: 't-card-marking',
  props: {
    label: String,
    type: String,
  },
  computed: {
  },
};
</script>

<style lang="scss" scoped>
.t-card-marking {
  
}
</style>