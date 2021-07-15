<template>
  <div class="board">
    <div class="graphics">
      <div class="wrapper">
        <div class="tabs-content">
          <div class="inner">
            <div class="tabs-item graphics">
              <div
                class="mCustomScrollBox mCS-light mCSB_vertical mCSB_outside"
              >
                <div class="mCSB_container">
                  <div class="tab-container">
                    <Chars v-if="chars" />
                    <Scatters  v-if="scatters" />
                    <Images  v-if="images" />
                    <Texts  v-if="texts" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters } from "vuex";
import Images from "./main/images/Images.vue";
import Texts from "./main/texts/Texts.vue";
import Scatters from "./main/Scatters.vue";
import Chars from "./main/Chars.vue";

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
    }),
  },
};
</script>

<style scoped>
</style>