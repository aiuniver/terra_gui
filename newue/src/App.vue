<template>
  <div class="app" data-theme="light">
    <div class="app__header">
      <Header />
    </div>
    <div class="app__main">
      <router-view />
    </div>
    <div class="app_footer">
      <Footer />
    </div>
  </div>
</template>

<style lang="scss">
.app {
  height: 100%;
  width: 100%;
  &__header {
  }
  &__main {
  }
  &__footer {
  }
}
</style>

<script>
import Header from '@/components/app/Header';
import Footer from '@/components/app/Footer';
export default {
  name: 'app',
  components: {
    Header,
    Footer,
  },
  async created() {
    await this.$store.dispatch('init');
  },
};
</script>