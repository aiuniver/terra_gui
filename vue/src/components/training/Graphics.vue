<template>
  <div class="board">
    <scrollbar :style="height">
      <div class="tab-container">
        <Chars v-if="chars" />
        <Scatters v-if="scatters" />
        <Images v-if="images" />
        <Texts v-if="texts" />
      </div>
    </scrollbar>
  </div>
</template>

<script>
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
  },
  computed: {
    ...mapGetters({
      chars: "trainings/getToolbarChars",
      scatters: "trainings/getToolbarScatters",
      images: "trainings/getToolbarImages",
      texts: "trainings/getToolbarTexts",
      height: "settings/autoHeight",
    }),
  },
};
</script>

<style scoped>
.tab-container {
  padding: 20px;
  display: flex;
  flex-direction: column;
  flex-wrap: wrap;
}
</style>