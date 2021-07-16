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
import { mapGetters } from "vuex";
// import Message from "@/components/app/Message";
import Footer from "@/components/app/Footer";
// import Bar from "@/components/app/Bar";

export default {
  name: "App",
  components: {
    Header,
    Nav,
    Footer
  },
  computed: {
    ...mapGetters({
      message: "messages/getText",
    }),
  },
  created() {
    this.$store.dispatch("datasets/get");
    // this.$Message('Thanks for using AT-UI')
  },
};
</script>

