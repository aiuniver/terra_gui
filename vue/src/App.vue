<template>
  <v-app id="app">
    <Bar />
    <Message />
    <v-main>
      <v-container fluid>
        <router-view></router-view>
      </v-container>
    </v-main>

    <Footer />
  </v-app>
</template>

<script>
import { mapGetters } from "vuex";
import Message from "@/components/app/Message";
import Footer from "@/components/app/Footer";
import Bar from "@/components/app/Bar";

export default {
  name: "App",
  components: {
    Message,
    Footer,
    Bar,
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

