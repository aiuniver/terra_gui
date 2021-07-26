<template>
  <div>
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
    Footer
  },
  async created() {
    await this.$store.dispatch("datasets/get");
    await this.$store.dispatch("projects/get");
    if (!this.$store.state.projects.project.dataset && this.$route.path !== '/datasets') {
      this.$router.push('/datasets');
    }
  },
};
</script>

