<template>
  <div class="t-charts">
    <!-- <div class="charts__title">Графики</div> -->
    <div class="t-charts__content">
        <TCharTemp class="t-chart" />
        <TChar class="t-chart" v-for="(char, i) of lossGraphs" :key="'char1_' + i" :char="char" :index="i" />
    </div>
  </div>
</template>

<script>
import TChar from './TChar';
import TCharTemp from './TCharTemp';
import { mapGetters } from 'vuex';
export default {
  name: 'TLossGraphs',
  components: {
    TChar,
    TCharTemp,
  },
  props: {
    metric: String
  },
  computed: {
    ...mapGetters({
      chars: 'trainings/getChars',
    }),
    lossGraphs() {
      return this.$store.getters['trainings/getTrainData'](this.metric) || [];
    },
  },
  mounted() {
    this.$emit('isLoad', true);
    console.log('mounted');
  },
};
</script>

<style lang="scss" scoped>
.t-chart {
  width: 48%;
  margin: 0 0 20px 0;
}
.t-charts {
  margin-bottom: 20px;

  // &__title {
  // }
  &__content {
    margin-top: 10px;
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    justify-content: flex-start;
    align-content: flex-start;
    align-items: flex-start;
    gap: 2%;
  }
}
</style>