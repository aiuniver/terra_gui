<template>
  <div class="row">
    <div class="col-24">
      <div class="row">
        <div class="col-4">Test</div>
        <div class="col-20">Test</div>
      </div>
    </div>
    <div class="col-8 d-flex"><CardFormInput /></div>
    <div class="col-16">test</div>
  </div>
</template>

<script>
import { mapGetters } from "vuex";
import CardFormInput from '@/components/datasets/CardFormInput.vue'
// /home/bondrogeen/github/terra/vue/src/components/datasets/CardFormInput.vue
// import serialize from "@/assets/js/serialize";
export default {
  name: "Settings",
  components: {
    CardFormInput
  },
  data: () => ({}),
  computed: {
    ...mapGetters({
      settings: "datasets/getSettings",
    }),
  },
};
</script>

<style>
.full {
  width: 1000px;
}
</style>