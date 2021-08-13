<template>
  <main class="page-datasets">
    <div class="cont">
      <Dataset v-show="!full" />
      <ParamsFull v-show="full" />
      <Params v-show="!full" />
    </div>
  </main>
</template>

<script>
import { mapGetters } from 'vuex'
import Dataset from "@/components/datasets/Dataset.vue";
import Params from "@/components/datasets/params/Params.vue";
import ParamsFull from "@/components/datasets/paramsFull/ParamsFull.vue";

export default {
  name: "Datasets",
  components: {
    Dataset,
    Params,
    ParamsFull,
  },
  data: () => ({
  }),
  computed: {
    ...mapGetters({
      full: 'datasets/getFull'
    })
  }
};
</script>

<style lang="scss" scoped>
.cont{
  background-color: #0e1621;
  padding: 0;
  display: flex;
  height: 100%;
}

</style>