<template>
  <div class="page-create">
    <div class="page-create__toolbar">
      <Toolbar @action="onToolbar" />
    </div>
    <div class="page-create__main">
      <Blocks />
    </div>
    <div class="page-create__params">
      <Params v-model="state" />
    </div>
  </div>
</template>

<script>
import Toolbar from '@/components/datasets/create/toolbar/Toolbar';
import Blocks from '@/components/datasets/create/main';
import Params from '@/components/datasets/create/params';
export default {
  name: 'Datasets',
  components: {
    Toolbar,
    Blocks,
    Params,
  },
  data: () => ({
    state: 1
  }),
  methods: {
    onToolbar(action) {
      console.log(action);
    },
  },
};
</script>

<style lang="scss">
.page-create {
  position: relative;
  display: flex;
  height: 100%;
  &__toolbar {
  }
  &__main {
    flex: 1 1 auto;
  }

  &__params {
    flex: 0 0 550px;
    border-left: 1px solid #000;
  }
}
</style>