<template>
  <main class="page-choice">
    <div class="page-choice__menu">
      <ChoiceMenu />
    </div>
    <div class="page-choice__main">
      <Datasets :datasets="datasets" :selectedType="selectedType" />
    </div>
  </main>
</template>

<script>
import { mapGetters } from 'vuex';
import Datasets from '@/components/datasets/choice/';
import ChoiceMenu from '@/components/datasets/choice/ChoiceMenu';
export default {
  name: 'Choice',
  components: {
    Datasets,
    ChoiceMenu,
  },
  data: () => ({
    selectedType: 2,
  }),
  computed: {
    ...mapGetters({
      datasets: 'datasets/getDatasets',
      choiceDataset: 'datasets/choiceDataset',
    }),
  },
};
</script>

<style lang="scss">
.page-choice {
  position: relative;
  display: flex;
  &__menu {
    width: 175px;
    border-right: 1px solid $color-black;
    padding-top: 18px;
    hr {
      border: none;
      border-bottom: 1px solid $color-dark-gray;
      margin: 20px;
    }
  }
}
</style>