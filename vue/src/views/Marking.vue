<template>
  <main class="page-marking">
    <div class="cont">
      <MarkingPage />
      <Params />
    </div>
  </main>
</template>

<script>
export default {
  name: "Marking",
  components: {
    MarkingPage: () => import('@/components/marking/MarkingPage.vue'),
    Params: () => import('@/components/marking/params/Params.vue'),
  },
  data: () => ({
  }),
  computed: {
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