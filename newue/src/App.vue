<template>
  <div class="app" data-theme="light">
    <router-view />
  </div>
</template>

<style lang="scss">
.app {
  height: 100%;
  width: 100%;
}
</style>

<script>
export default {
  name: 'app',
  async created() {
    await this.$store.dispatch('projects/get');
  },
};
</script>