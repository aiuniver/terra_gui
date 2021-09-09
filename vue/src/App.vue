<template>
  <div class="app">
    <Overlay v-if="$store.state.settings.overlay" />
    <Header />
    <Nav />
    <router-view></router-view>
    <Footer />
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import Header from '@/components/app/Header';
import Nav from '@/components/app/Nav';
import Footer from '@/components/app/Footer';
import Overlay from './components/forms/Overlay';

export default {
  name: 'App',
  components: {
    Header,
    Nav,
    Footer,
    Overlay,
  },
  computed: {
    ...mapGetters({
      project: 'projects/getProject',
    })
  },
  methods: {
    myEventHandler() {
      const height = this.$el.clientHeight;
      const wigth = this.$el.clientWidth;
      this.$store.dispatch('settings/setResize', { height, wigth });
    },
  },
  async created() {
    await this.$store.dispatch('projects/get');
    await this.$store.dispatch('datasets/get');
    if (!this.project?.dataset) {
      console.log(this.$route);
      if (this.$route.meta.access == false) {
        try {
          const data = await this.$Modal.alert({
            title: 'Предупреждение!',
            width: 300,
            content: this.$route.meta.text,
            showClose: false,
            okText: 'Загрузить датасет',
          });
          if (data) {
            if (this.$route.path !== '/datasets') {
              this.$router.push('/datasets');
            }
          }
        } catch (error) {
          console.log(error)
        }
      }
    }
    window.addEventListener('resize', this.myEventHandler);
  },
  destroyed() {
    window.removeEventListener('resize', this.myEventHandler);
  },
};
</script>

<style lang="scss" scoped>
.app {
  height: 100%;
}
</style>

