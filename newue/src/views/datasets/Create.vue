<template>
  <main class="page page-datasets">
    <div class="page-datasets__toolbar">
      <Toolbar @action="handleWorkspaceAction" />
    </div>
    <div class="page-datasets__main">
      <Blocks />
    </div>
    <div class="page-datasets__params">
      <Params />
    </div>
  </main>
</template>

<script>
import Toolbar from '@/components/datasets/components/create/toolbar/Toolbar';
import Blocks from '@/components/datasets/components/create/main';
import Params from '@/components/datasets/components/create/params/Params';
export default {
  name: 'Datasets',
  components: {
    Toolbar,
    Blocks,
    Params,
  },

  methods: {
    handleWorkspaceAction(action) {
      console.log(action);
    },
  },
};
</script>

<style lang="scss">
.page-datasets {
  position: relative;
  display: flex;

  &__toopbar {
  }

  &__main {
    flex: 1 1 auto;
  }

  &__params {
    flex: 0 0 600px;
    border-left: 1px solid #000;
  }
}
</style>