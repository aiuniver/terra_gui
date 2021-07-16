<template>
  <div class="board">
    <vue-custom-scrollbar
      class="scroll-area"
      :settings="{
        suppressScrollY: false,
        suppressScrollX: true,
        wheelPropagation: false,
      }"
    >
      <div class="tab-container">
        <Chars v-if="chars" />
        <Scatters v-if="scatters" />
        <Images v-if="images" />
        <Texts v-if="texts" />
      </div>
    </vue-custom-scrollbar>
  </div>
</template>

<script>
import vueCustomScrollbar from "vue-custom-scrollbar";
import "vue-custom-scrollbar/dist/vueScrollbar.css";
import { mapGetters } from "vuex";
import Images from "./main/images/index.vue";
import Texts from "./main/texts/index.vue";
import Scatters from "./main/Scatters.vue";
import Chars from "./main/chars/index.vue";

export default {
  name: "Graphics",
  components: {
    Images,
    Texts,
    Scatters,
    Chars,
    vueCustomScrollbar,
  },
  computed: {
    ...mapGetters({
      chars: "trainings/getToolbarChars",
      scatters: "trainings/getToolbarScatters",
      images: "trainings/getToolbarImages",
      texts: "trainings/getToolbarTexts",
    }),
  },
};
</script>

<style scoped>
.scroll-area {
  position: relative;
  width: 100%;
  height: 800px;
}
.tab-container {
  padding: 20px;
  display: flex;
  flex-direction: column;
  flex-wrap: wrap;
}
</style>