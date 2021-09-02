<template>
  <div class="params">
    <scrollbar>
      <div class="params__items">
        <at-collapse :value="collapse">
          <at-collapse-item class="mt-3" :title="main.name">
            <template v-for="(data, i) of main.fields">
              <t-auto-field-trainings v-bind="data" :key="'main_' + i" :inline="false" @change="change" />
            </template>
          </at-collapse-item>
          <at-collapse-item class="mt-3" :title="fit.name">
            <template v-for="(data, i) of fit.fields">
              <t-auto-field-trainings v-bind="data" :key="'fit_' + i" :inline="true" @change="change" />
            </template>
          </at-collapse-item>
          <at-collapse-item class="mt-3" :title="optimizer.name">
            <template v-for="(data, i) of optimizerFields">
              <t-auto-field-trainings v-bind="data" :key="'optimizer_' + i" inline @change="change" />
            </template>
          </at-collapse-item>
          <at-collapse-item class="mt-3" :title="outputs.name">
            <template v-for="(data, i) of outputs.fields">
              <t-auto-field-trainings v-bind="data" :key="'outputs_' + i" :inline="true" @change="change" />
            </template>
          </at-collapse-item>
          <at-collapse-item class="mt-3" :title="checkpoints.name">
            <template v-for="(data, i) of checkpoints.fields">
              <t-auto-field-trainings v-bind="data" :key="'checkpoints_' + i" :inline="true" @change="change" />
            </template>
          </at-collapse-item>
        </at-collapse>
      </div>
      <div class="params__items--item">
        <div class="item d-flex mb-5" style="gap: 10px">
          <button>Обучить</button>
          <button>Остановить</button>
        </div>
        <div class="item d-flex" style="gap: 10px">
          <button>Сбросить</button>
          <button>Сбросить</button>
        </div>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import temp from './temp';
import { mapGetters } from 'vuex';
// import Select from '@/components/forms/Select.vue';
// import Checkbox from '@/components/forms/Checkbox.vue';

export default {
  name: 'params-traning',
  components: {
    // Select,
    // Checkbox,
  },
  data: () => ({
    collapse: [0, 1, 2, 3, 4],
    temp,
    optimizerValue: '',
  }),
  computed: {
    ...mapGetters({
      params: 'trainings/getParams',
    }),
    main() {
      return this.params.main;
    },
    fit() {
      return this.params.fit;
    },
    outputs() {
      return this.params.outputs
    },
    optimizerFields() {
      return this.params.optimizer.fields[this.optimizerValue];
    },
    optimizer() {
      return this.params.optimizer;
    },
    checkpoints() {
      return this.params.checkpoints
    },
  },
  methods: {
    change(e) {
      console.log(e);
      if (e.name === 'optimizer') {
        this.optimizerValue = e.value;
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.params {
  width: 400px;
  flex-shrink: 0;
  border-left: #0e1621 solid 1px;
  overflow: hidden;
  height: 100%;
  // border-left: #0e1621  1px solid;
  &__items {
    height: 100%;
    padding-bottom: 20px;
    &--item {
      padding: 20px;
    }
  }
}
</style>