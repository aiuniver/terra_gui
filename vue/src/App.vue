<template>
  <div class="app" ref="terra">
    <Header />
    <Nav />
    <router-view></router-view>
    <Footer />
  </div>
</template>

<script>
import Header from "@/components/app/Header.vue";
import Nav from "@/components/app/Nav.vue";
import Footer from "@/components/app/Footer";

export default {
  name: "App",
  components: {
    Header,
    Nav,
    Footer,
  },
  data: () => ({}),
  methods: {
    myEventHandler() {
      const height = this.$refs.terra.clientHeight;
      const wigth = this.$refs.terra.clientWidth;
      this.$store.dispatch("settings/setResize", { height, wigth });
    },
  },
  async created() {
    await this.$store.dispatch("projects/get");
    await this.$store.dispatch("datasets/get");
    if (
      !this.$store?.state?.projects?.project?.dataset &&
      this.$route.path !== "/datasets"
    ) {
      const text = {
        "/modeling": "редактирования модели",
        "/training": "обучения",
        "/deploy": "деплоя",
      };
      const self = this;
      this.$Modal.alert({
        title: "Предупреждение!",
        width: 300,
        content: `Для ${text[this.$route.path]} необходимо загрузить датасет.`,
        showClose: false,
        okText: "Загрузить датасет",
        callback: function () {
          self.$router.push("/datasets");
        },
      });
    }
    // window.addEventListener("resize", this.myEventHandler);
  },
  destroyed() {
    // window.removeEventListener("resize", this.myEventHandler);
  },
};
</script>

<style lang="scss" scoped>
.app {
  height: 100%;
}
</style>

