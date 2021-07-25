<template>
  <main class="page-datasets">
    <div class="cont">
      <Dataset />
      <ParamsFull v-if="full" />
      <Params v-else />
    </div>
  </main>
</template>

<script>
import { mapGetters } from 'vuex'
import Dataset from "@/components/datasets/Dataset.vue";
import Params from "@/components/datasets/params/Params.vue";
import ParamsFull from "@/components/datasets/params/ParamsFull.vue";

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
  background: #17212B;
  padding: 0;
  display: flex;
}

</style>