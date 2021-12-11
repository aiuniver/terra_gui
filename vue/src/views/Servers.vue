<template>
	<main class="page-servers">
		<div class="page-servers__list">
		<scrollbar class="page-servers__scroll">
			<div class="page-servers__table-wrapper">
				<ServerTable v-if="showTable" @instruction="openInstruction" :servers="servers" />
				<p v-else class="page-servers__noserver">Нет добавленных серверов демо-панелий</p>
				<span class="page-servers__btn" @click="addNew = true">
					<i class="ci-icon ci-plus_circle"></i><span>Добавить сервер</span>
				</span>
			</div>
		</scrollbar>
		</div>
		<div class="page-servers__new">
			<NewServer v-show="addNew" @addServer="newServer" />
		</div>
		<at-modal v-model="serverModal" class="modal" okText="Читать инструкцию" @on-confirm="openInstruction(serverID)">
			<template v-slot:header><span class="modal-title">Сервер демо-панели добавлен</span></template>
			<p>Ознакомьтесь с дальнейшими действиями в <span class="clickable" @click="openInstruction(serverID)">Инструкции</span></p>
			<p>Вы также сможете найти ее в таблице серверов на владке Серверы демо-панелей в вашем Профиле</p>
		</at-modal>
		<at-modal v-model="InstructionModal" class="modal">
			<template v-slot:header><span class="modal-title">Инструкция по настройке сервера демо-панели</span></template>
			<t-field label="Приватный SSH-ключ">
				<div class="ssh-wrapper">
					<span class="ssh" ref="private">{{ private_key }}</span> <i title="Скопировать" @click="copy($refs.private)" class="btn-copy"></i> <span class="clickable">Скачать</span>
				</div>
			</t-field>
			<t-field label="Публичный  SSH-ключ">
				<div class="ssh-wrapper">
					<span class="ssh" ref="public">{{ public_key }}</span> <i title="Скопировать" @click="copy($refs.public)" class="btn-copy"></i> <span class="clickable">Скачать</span>
				</div>
			</t-field>
		</at-modal>
	</main>
</template>

<script>
import ServerTable from '@/components/servers/ServerTable.vue'
import NewServer from '@/components/servers/NewServer.vue'
import { mapGetters } from 'vuex'

export default {
	name: 'servers',
	components: {
		ServerTable,
		NewServer
	},
	data: () => ({
		addNew: false,
		serverModal: false,
		InstructionModal: false,
		serverID: null,
		private_key: null,
		public_key: null
	}),
	computed: {
		...mapGetters({
      servers: 'servers/getServers'
    }),
		showTable() {
			return !!Object.keys(this.$store.getters['servers/getServers']).length
		},
		getServer() {
			return this.servers.find(server => server.id === this.serverID)
		}
	},
	methods: {
		openManual(server) {
			this.selectedServer = server
			this.manualModal = true
		},
		copy(ref) {
			let selection = window.getSelection()
      let range = document.createRange()

      range.selectNodeContents(ref)
      selection.removeAllRanges()
      selection.addRange(range)
      document.execCommand('copy')
      selection.removeAllRanges()
		},
		newServer(id) {
			this.serverID = id
			this.serverModal = true
		},
		async openInstruction(id) {
			this.serverID = id
			this.serverModal = false
			this.InstructionModal = true
			const { data } = await this.$store.dispatch('servers/getInstruction', { id })
			this.private_key = data.private_ssh_key
			this.public_key = data.public_ssh_key
		}
	},
	created() {
    this.$store.dispatch('servers/getServers')
  }
}
</script>

<style lang="scss" scoped>
.page-servers {
	display: flex;
	height: 100%;
	&__list {
		height: 100%;
		background: #17212B;
		flex-grow: 1;
	}
	&__table-wrapper {
		padding: 30px 20px;
	}
	&__noserver {
		font-size: 12px;
		color: #A7BED3;
	}
	&__btn {
		color: #65B9F4;
		cursor: pointer;
		font-size: 14px;
		display: flex;
		align-items: center;
		gap: 10px;
		width: max-content;
		margin-top: 40px;
		i {
			font-size: 20px;
		}
	}
	&__new {
		flex: 0 0 350px;
		border-left: #0e1621 1px solid;
		background: #17212B;
	}
}

.modal {
	&-title {
		font-size: 16px;
	}
	p {
		color: #A7BED3;
		font-size: 14px;
		margin-bottom: 20px;
	}
	.clickable {
		color: #65B9F4;
		cursor: pointer;
		position: relative;
		&:hover::after {
			content: "";
			position: absolute;
			bottom: -5px;
			left: 0;
			width: 100%;
			height: 1px;
			background: #65B9F4;
		}
	}
}

.btn-copy {
  display: inline-block;
  vertical-align: middle;
  cursor: pointer;
  width: 24px;
  height: 24px;
  background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iI0E3QkVEMyI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xNiAxSDRjLTEuMSAwLTIgLjktMiAydjE0aDJWM2gxMlYxem0zIDRIOGMtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxMWMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0wIDE2SDhWN2gxMXYxNHoiLz48L3N2Zz4=');
}

.ssh {
	text-overflow: ellipsis;
	overflow: hidden;
	max-width: 300px;
	white-space: nowrap;
	&-wrapper {
		display: flex;
		align-items: center;
		gap: 20px;
	}
}
</style>