<template>
	<at-modal v-model="dialog" width="400" showClose okText="Выбрать" @on-confirm="confirm">
		<template v-slot:header>Запуск</template>
		<div class="t-modal-datasets">
			<template v-for="(block, idx) in inputBlocks">
				<t-field :label="block.name" :key="idx">
					<t-select-new small :list="datasets" style="width: 239px;" v-model="selected[block.id]"/>
				</t-field>
			</template>
		</div>	
	</at-modal>
</template>

<script>
import { mapGetters } from 'vuex'
import TSelectNew from '../comp/TSelect.vue'

export default {
	name: 'ModalDatasets',
	components: {
		TSelectNew
	},
	props: {
		value: Boolean
	},
	data: () => ({
		selected: {}
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
			datasets: 'cascades/getDatasets'
		}),
		inputBlocks() {
			return this.getBlocks.filter(item => item.group === 'InputData')
		}
	},
	methods: {
		async confirm() {
			await this.$store.dispatch('cascades/start', this.selected)
			this.createInterval()
		},
		createInterval() {
      this.interval = setTimeout(async () => {
				this.$store.dispatch('settings/setOverlay', true);
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
                this.$store.dispatch('messages/setMessage', { error });
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
	}
}
</script>

<style lang="scss" scoped>
.t-modal-datasets {
	display: flex;
	flex-direction: column;
	gap: 10px;
}
</style>