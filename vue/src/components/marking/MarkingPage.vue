<template>
  <div class="board">
    <div class="board__inner">
      <div :class="['board__files', { toggle: !toggle }]">
        <BlockFiles @toggle="change" />
      </div>
      <div class="board__main">
        <t-card-marking />
        <t-card-images />
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import BlockFiles from '@/components/marking/params/block/BlockFiles.vue';
import tCardMarking from '@/components/marking/params/components/card/CardMarking.vue';
import tCardImages from '@/components/marking/params/components/card/CardImages.vue';
export default {
  components: {
    tCardMarking,
    tCardImages,
    BlockFiles,
  },
  data: () => ({
    hight: 0,
    toggle: false,
  }),
  computed: {
    ...mapGetters({}),
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
  }
}
</style>
