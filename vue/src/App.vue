<template>
  <div>
    <Header />
    <Nav />
    <main class="page-datasets">
      <div class="container">
        <div class="board">
          <div class="wrapper">
            <router-view></router-view>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import Header from "@/components/app/Header.vue";
import Nav from "@/components/app/Nav.vue";
import { mapGetters } from "vuex";
// import Message from "@/components/app/Message";
// import Footer from "@/components/app/Footer";
// import Bar from "@/components/app/Bar";

export default {
  name: "App",
  components: {
    Header,
    Nav,
  },
  computed: {
    ...mapGetters({
      message: "messages/getText",
    }),
  },
  created() {
    this.$store.dispatch("datasets/get");
  },
};
</script>

