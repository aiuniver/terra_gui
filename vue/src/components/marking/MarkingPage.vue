<template>
  <div class="board">
    <div class="board__inner">
      <div v-if="files.length > 0" :class="['board__files', { toggle }]">
        <BlockFiles @toggle="toggle = !toggle" />
      </div>
      <div v-else class="board__toolbar">
        <Toolbar />
      </div>
      <div class="board__main">
        <ImageCards v-if="files.length > 0" />
        <MarkingCards v-else />
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import BlockFiles from '@/components/marking/params/block/BlockFiles.vue';
import MarkingCards from '@/components/marking/params/block/MarkingCards.vue';
import ImageCards from '@/components/marking/params/block/ImageCards.vue';
import Toolbar from './params/Toolbar.vue'

export default {
  components: {
    BlockFiles,
    Toolbar,
    MarkingCards,
    ImageCards
  },
  data: () => ({
    hight: 0,
    toggle: false,
  }),
  computed: {
    ...mapGetters({
      files: 'datasets/getFilesSource'
    }),
  },
  methods: {
    change(e) {
      console.log(e);
    },
  },
};
</script>

<style lang="scss" scoped>
.board {
  flex-shrink: 0;
  width: 100%;
  // padding-left: 41px;
  flex: auto;
  display: -webkit-flex;
  display: flex;
  background-color: #0e1621;
  border-top: #0e1621 solid 1px;
  &__inner {
    width: 100%;
    display: -webkit-flex;
    display: flex;
    position: relative;
    background-color: #17212b;
  }
  &__files {
    flex: 0 0 190px;
    display: -webkit-flex;
    display: flex;
    border-right: #0e1621 solid 1px;
    &.toggle {
      flex: 0 0 40px;
    }
  }
  &__main {
    flex: 1 1;
    display: -webkit-flex;
    display: flex;
    width: 100%;
    overflow: hidden;
    flex-direction: column;
  }
  &__toolbar {
    height: 100%;
  }
}
</style>
