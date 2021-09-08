<template>
  <main class="page-training">
    <div class="container">
      <Toolbar/>
      <Graphics/>
      <Params/>
    </div>
  </main>
</template>

<script>
// import { mapGetters } from "vuex";
import Toolbar from "@/components/training/Toolbar.vue";
import Graphics from "@/components/training/Graphics.vue";
import Params from "@/components/training/Params.vue";

export default {
  name: "Training",
  components: {
    Toolbar,
    Graphics,
    Params
  },
  data() {
    return {

    };
  },
  methods: {

  },
  created() {
    // console.log(this.$store.dispatch('trainings/data'))
  }
};
</script>

