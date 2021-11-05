<template>
  <div class="app" data-theme="light">
    <Header />
    <div class="app__main">
      <router-view />
    </div>
    <Footer />
  </div>
</template>

<style lang="scss">
.app {
  height: 100%;
  width: 100%;
  &__header{

  }
  &__main {

  }
  &__footer{

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
    await this.$store.dispatch('themes/setTheme');
    await this.$store.dispatch('projects/get');
  },
};
</script>