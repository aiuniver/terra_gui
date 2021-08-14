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
export default {
  name: "Datasets",
  components: {
    Dataset: () => import('@/components/datasets/Dataset.vue'),
    Params: () => import('@/components/datasets/params/Params.vue'),
    ParamsFull: () => import('@/components/datasets/paramsFull/ParamsFull.vue'),
  },
  data: () => ({
  }),
  computed: {
    full() {
      return this.$store.getters['datasets/getFull']
    }
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