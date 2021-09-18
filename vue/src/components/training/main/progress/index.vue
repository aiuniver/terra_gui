<template>
  <div class="t-progress">
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

export default {
  name: 't-progress',
  components: {
    Sysinfo,
    Timers,
  },
  computed: {
    lossGraphs() {
      return this.$store.getters['trainings/getTrainUsage'] || {};
    },
    usage() {
      return this.lossGraphs?.hard_usage || {};
    },
    timings() {
      return this.lossGraphs?.timings || {};
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
  &__item {

    &--timers {
    }
    &--info {
      width: 40%;
    }
  }
}
</style>
