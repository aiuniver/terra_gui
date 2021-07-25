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
  created() {
    this.$store.dispatch("datasets/get");
    this.$store.dispatch("settings/get");
  },
};
</script>

