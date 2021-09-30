<template>
  <div class="t-progress">
    <div class="t-progress__overlay" v-if="statusTrain === 'start' && !isEmpty">
      <LoadSpiner text="Загрузка данных..." />
    </div>
    <div class="t-progress__item t-progress__item--timers">
      <Timers v-bind="timings" />
    </div>
    <div class="t-progress__item t-progress__item--info">
      <Sysinfo v-bind="usage" />
    </div>
  </div>
</template>

<script>
import Sysinfo from './Sysinfo.vue';
import Timers from './Timers.vue';
import LoadSpiner from '@/components/forms/LoadSpiner';
import { mapGetters } from 'vuex';

export default {
  name: 't-progress',
  components: {
    Sysinfo,
    Timers,
    LoadSpiner,
  },
  computed: {
    ...mapGetters({
      status: 'trainings/getStatus',
    }),
    isLearning() {
      return ['addtrain', 'training'].includes(this.status);
    },
    isEmpty() {
      console.log(Object.keys(this.lossGraphs).length);
      return Object.keys(this.lossGraphs).length;
    },
    lossGraphs() {
      return this.$store.getters['trainings/getTrainUsage'] || {};
    },
    usage() {
      return this.lossGraphs?.hard_usage || {};
    },
    timings() {
      return this.lossGraphs?.timings || {};
    },
    statusTrain() {
      return this.$store.getters['trainings/getStatusTrain'];
    },
  },
  mounted() {
    console.log(this.data);
  },
};
</script>

<style lang="scss" scoped>
.t-progress {
  padding: 10px 0;
  font-size: 12px;
  line-height: 24px;
  color: #a7bed3;
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 5px;
  position: relative;
  &__overlay {
    position: absolute;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    background-color: rgb(14 22 33 / 30%);
    z-index: 5;
  }
  &__item {
    &--info {
      display: flex;
      align-items: flex-end;
      width: 30%;
    }
  }
}
</style>
