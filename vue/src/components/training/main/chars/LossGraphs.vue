<template>
  <div class="charts">
    <div class="charts__title">Графики</div>
    <div class="charts__content">
      <TChar v-for="(char, i) of lossGraphs" :key="'char1_' + i" :char="char" />
      <TCharTemp />
    </div>
  </div>
</template>

<script>
import TChar from './TChar.vue';
import TCharTemp from './TCharTemp.vue';
import { mapGetters } from 'vuex';
export default {
  name: 'TLossGraphs',
  components: {
    TChar,
    TCharTemp
  },
  data: () => ({
    
  }),
  computed: {
    ...mapGetters({
      chars: 'trainings/getChars',
    }),
    lossGraphs() {
      return this.$store.getters['trainings/getTrainData']('loss_graphs') || [];
    },
  },
  mounted() {
    this.$emit('isLoad', true);
    console.log('mounted');
  },
};
</script>

<style lang="scss" scoped>
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
    -webkit-box-direction: normal;
    -moz-box-direction: normal;
    -webkit-box-orient: horizontal;
    -moz-box-orient: horizontal;
    -webkit-flex-direction: row;
    -ms-flex-direction: row;
    flex-direction: row;
    -webkit-flex-wrap: wrap;
    -ms-flex-wrap: wrap;
    flex-wrap: wrap;
    -webkit-box-pack: start;
    -moz-box-pack: start;
    -webkit-justify-content: flex-start;
    -ms-flex-pack: start;
    justify-content: flex-start;
    -webkit-align-content: flex-start;
    -ms-flex-line-pack: start;
    align-content: flex-start;
    -webkit-box-align: start;
    -moz-box-align: start;
    -webkit-align-items: flex-start;
    -ms-flex-align: start;
    align-items: flex-start;
  }
}
</style>