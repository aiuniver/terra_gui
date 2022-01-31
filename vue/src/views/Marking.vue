<template>
  <div class="page-marking">
    <MarkingPage />
    <Params />
  </div>
</template>

<script>
export default {
  name: 'Marking',
  components: {
    MarkingPage: () => import('@/components/marking/MarkingPage.vue'),
    Params: () => import('@/components/marking/params/Params.vue'),
  },
  data: () => ({}),
  computed: {},
};
</script>

<style lang="scss" scoped>
.page-marking {
  background-color: #0e1621;
  padding: 0;
  display: flex;
  height: 100%;
}
</style>