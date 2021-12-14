<template>
  <at-modal
    v-model="dialog"
    width="400"
    okText="Выбрать"
    @on-confirm="confirm"
    title="Запуск"
    :showConfirmButton="isReady"
  >
    <div class="t-modal-datasets">
      <template v-for="(block, idx) in inputBlocks">
        <t-field :label="block.name" :key="idx">
          <t-auto-complete-new :list="filters" placeholder="Выберите датасет" @change="change(block.id, $event)" />
          <!-- <t-select-new small :list="datasets" v-model="selected[block.id]" placeholder="Выберите датасет" @change="change"/> -->
        </t-field>
      </template>
    </div>
  </at-modal>
</template>

<script>
import { mapGetters } from 'vuex';
// import TSelectNew from '../comp/TSelect.vue';

export default {
  name: 'ModalDatasets',
  components: {
    // TSelectNew,
  },
  props: {
    value: Boolean,
  },
  data: () => ({
    selected: {},
  }),
  computed: {
    dialog: {
      set(value) {
        this.$emit('input', value);
      },
      get() {
        return this.value;
      },
    },
    ...mapGetters({
      getBlocks: 'cascades/getBlocks',
      datasets: 'cascades/getDatasets',
    }),
    filters() {
      return this.datasets.map(i => ({ label: i.label, value: i.alias }));
    },
    inputBlocks() {
      return this.getBlocks.filter(item => item.group === 'InputData');
    },
    isReady() {
      return Object.keys(this.selected).length === this.inputBlocks.length;
    },
  },
  methods: {
    change(id, { value }) {
      const dataset = this.datasets.find(i => i.alias === value);
      if (dataset) {
        const { alias, group } = dataset;
        this.selected[id] = { alias, group };
        this.selected = { ...this.selected };
      }
    },
    async confirm() {
      if (!this.isReady) return this.$store.dispatch('messages/setMessage', { error: 'Выберите датасеты' });
      this.$store.dispatch('settings/setOverlay', true);
      await this.$store.dispatch('cascades/start', this.selected);
      this.createInterval();
    },
    createInterval() {
      this.interval = setTimeout(async () => {
        const res = await this.$store.dispatch('cascades/startProgress');
        if (res) {
          const { data } = res;
          if (data) {
            const { finished, message, percent, error } = data;
            this.$store.dispatch('messages/setProgressMessage', message);
            this.$store.dispatch('messages/setProgress', percent);
            if (finished) {
              this.$store.dispatch('messages/setProgress', 0);
              this.$store.dispatch('messages/setProgressMessage', '');
              await this.$store.dispatch('projects/get');
              this.$store.dispatch('settings/setOverlay', false);
            } else {
              if (error) {
                // this.$store.dispatch('messages/setMessage', { error });
                this.$store.dispatch('messages/setProgressMessage', '');
                this.$store.dispatch('messages/setProgress', 0);
                this.$store.dispatch('settings/setOverlay', false);
                return;
              }
              this.createInterval();
            }
          } else {
            this.$store.dispatch('settings/setOverlay', false);
          }
        } else {
          this.$store.dispatch('settings/setOverlay', false);
        }
      }, 1000);
    },
  },
};
</script>

<style lang="scss" scoped>
.t-modal-datasets {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
</style>