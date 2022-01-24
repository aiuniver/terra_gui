<template>
  <div class="page-marking">
    <div class="page-marking__files">

    </div>
    <div class="page-marking__toolbar">
      <!-- <Toolbar /> -->
    </div>
    <div class="page-marking__main">
      <!-- <ImageCards /> -->
      <MarkingCards  />
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
import MarkingCards from '../block/MarkingCards.vue';
// import ImageCards from '@/components/marking/params/block/ImageCards.vue';
// import Toolbar from './params/Toolbar.vue';

export default {
  components: {
    // Toolbar,
    MarkingCards,
    // ImageCards,
  },
  data: () => ({
    hight: 0,
    toggle: false,
  }),
  computed: {
    ...mapGetters({
      files: 'datasets/getFilesSource',
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
.page-marking {
  display: flex;

  &__toolbar {
    // height: 100%;
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
}
</style>
