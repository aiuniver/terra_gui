<template>
  <main class="page-marking">
    <div class="page-marking__main">
      <MarkingPage />
    </div>
    <div class="page-marking__params">
      <Params />
    </div>
  </main>
</template>

<script>
import MarkingPage from '@/components/marking/main/';
import Params from '@/components/marking/params/';
export default {
  name: 'Marking',
  components: {
    MarkingPage,
    Params,
  },
  data: () => ({}),
  computed: {},
};
</script>

<style lang="scss">
.page-marking {
  display: flex;
  &__main {
    flex: 1 1 auto;
  }
  &__params {
    flex: 0 0 400px;
  }
}
</style>