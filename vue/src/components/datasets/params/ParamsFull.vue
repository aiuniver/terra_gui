<template>
  <div class="row params">
    <div class="col-24 params__top">
      <div class="row">
        <div class="col-4 params__top--left">Test</div>
        <div class="col-20 params__top--rigth">
          <CardFile />
          <CardFile />
        </div>
      </div>
    </div>
    <div class="col-12 d-flex justify-end pa-3">
      <div class="row">
        <CardFormInput />
      </div>
    </div>
    <div class="col-12 pa-3">
      <div class="row">
      <CardFormInput />
      </div>

    </div>
  </div>
</template>

<script>
import { mapGetters } from "vuex";
import CardFormInput from "@/components/datasets/CardFormInput.vue";
import CardFile from "@/components/datasets/CardFile.vue";
// /home/bondrogeen/github/terra/vue/src/components/datasets/CardFormInput.vue
// import serialize from "@/assets/js/serialize";
export default {
  name: "Settings",
  components: {
    CardFormInput,
    CardFile,
  },
  data: () => ({}),
  computed: {
    ...mapGetters({
      settings: "datasets/getSettings",
    }),
  },
};
</script>


<style lang="scss">
.full {
  width: 1000px;
}
@import "@/at-ui/scss/variables/index.scss";

.params {
  position: relative;

  &__top {
    padding: 10px;
    background-color: #1b242e;
    &--rigth {
      display: flex;
      justify-content: center;
      padding: 8px;
    }
  }
}
</style>