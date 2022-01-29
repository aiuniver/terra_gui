<template>
  <div :class="['app', { 'app--new': isNew }]">
    <Overlay v-if="$store.state.settings.overlay" />
    <header>
      <template v-if="isNew">
        <NewHeader />
      </template>
      <template v-else>
        <Header />
        <Nav />
      </template>
    </header>
    <main>
      <router-view></router-view>
    </main>
    <footer>
      <Footer />
    </footer>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import Header from '@/components/app/Header';
import NewHeader from '@/components/app/NewHeader';
import Nav from '@/components/app/Nav';
import Footer from '@/components/app/Footer';
import Overlay from './components/forms/Overlay';

export default {
  name: 'App',
  components: {
    Header,
    NewHeader,
    Nav,
    Footer,
    Overlay,
    // SaveProject: () => import('@/components/app/modal/SaveProject.vue'),
  },
  data: () => ({
    dialogSave: false,
  }),
  computed: {
    ...mapGetters({
      project: 'projects/getProject',
    }),
    isNew() {
      const path = this?.$route?.fullPath || '';
      return Boolean(path.includes('new'));
    },
  },
  methods: {
    myEventHandler() {
      const height = this.$el.clientHeight;
      const wigth = this.$el.clientWidth;
      this.$store.dispatch('settings/setResize', { height, wigth });
    },
    // handlerClose: async function () {
    //   try {
    //     const res = await this.$Modal.confirm({
    //       title: 'Предупреждение!',
    //       width: 300,
    //       content: 'Данные не сохранены.',
    //       showClose: false,
    //       okText: 'Сохранить',
    //     });
    //     if (res === 'confirm') {
    //       this.dialogSave = true;
    //     }
    //   } catch (error) {
    //     console.log(error)
    //   }
    // },
  },
  async created() {
    await this.$store.dispatch('projects/get');
    await this.$store.dispatch('datasets/get');
    if (!this.project?.dataset) {
      // console.log(this.$route);
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
          console.log(error);
        }
      }
    }
    window.addEventListener('resize', this.myEventHandler);
    // window.addEventListener('beforeunload', event => {
    //   this.handlerClose();
    //   event.returnValue = '';
    // });
  },
  destroyed() {
    window.removeEventListener('resize', this.myEventHandler);
  },
};
</script>

<style lang="scss">
.app {
  height: 100%;
  display: flex;
  flex-direction: column;
  header {
    flex: 0 0 92px;
  }
  main {
    height: calc(100% - 152px);
    flex: 1 1 auto;
  }
  footer {
    flex: 0 0 60px;
  }
  &--new {
    header {
      flex: 0 0 52px;
    }
    main {
      height: calc(100% - 112px);
      flex: 1 1 auto;
    }
  }
}
</style>

