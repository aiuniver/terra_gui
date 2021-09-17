<template>
  <div class="charts">
    <div class="charts__title">Графики</div>
    <div class="charts__content">
      <div class="chart">
        <TCharTemp />
      </div>
      <div v-for="(char, i) of lossGraphs" :key="'char1_' + i" class="chart">
        <TChar :char="char" />
      </div>
    </div>
  </div>
</template>

<script>
import TChar from './TChar';
import TCharTemp from './TCharTemp';
import { mapGetters } from 'vuex';
export default {
  name: 'TMetricGraphs',
  components: {
    TChar,
    TCharTemp,
  },
  computed: {
    ...mapGetters({
      chars: 'trainings/getChars',
    }),
    lossGraphs() {
      return this.$store.getters['trainings/getTrainData']('metric_graphs') || [];
    },
  },
  mounted() {
    this.$emit('isLoad', true);
    console.log('mounted');
  },
};
</script>

<style lang="scss" scoped>
.chart {
  width: 48%;
  margin: 0 0 20px 0;
}
.charts {
  margin-bottom: 20px;
  // &__title {
  // }
  &__content {
    margin-top: 10px;
    display: -webkit-box;
    display: -moz-box;
    display: -ms-flexbox;
    display: -webkit-flex;
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