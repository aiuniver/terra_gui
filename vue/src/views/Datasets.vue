<template>
  <div class="page-datasets">
    <Dataset v-show="!full" />
    <ParamsFull v-if="full" />
    <Params v-else />
  </div>
</template>

<script>
export default {
  name: 'Datasets',
  components: {
    Dataset: () => import('@/components/datasets/Dataset.vue'),
    Params: () => import('@/components/datasets/params/Params.vue'),
    ParamsFull: () => import('@/components/datasets/paramsFull/ParamsFull.vue'),
  },
  data: () => ({}),
  computed: {
    full() {
      return this.$store.getters['datasets/getFull'];
    },
  },
};
</script>

<style lang="scss" scoped>
.page-datasets {
  background-color: #0e1621;
  padding: 0;
  display: flex;
  height: 100%;
}
</style>