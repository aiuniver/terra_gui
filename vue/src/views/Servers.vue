<template>
	<main class="page-servers">
		<div class="page-servers__list">
			<ServerTable />
			<p class="page-servers__noserver">Нет добавленных серверов демо-панелий</p>
			<span class="page-servers__btn" @click="addNew = true">
				<i class="ci-icon ci-plus_circle"></i><span>Добавить сервер</span>
			</span>
		</div>
		<div class="page-servers__new">
			<NewServer v-show="addNew" @addserver="serverModal = true" />
		</div>
		<at-modal v-model="serverModal" class="modal" okText="Читать инструкцию">
			<template v-slot:header><span class="modal-title">Сервер демо-панели добавлен</span></template>
			<p>Ознакомьтесь с дальнейшими действиями в <span class="clickable" @click="manualModal = true; serverModal = false">Инструкции</span></p>
			<p>Вы также сможете найти ее в таблице серверов на владке Серверы демо-панелей в вашем Профиле</p>
		</at-modal>
		<at-modal v-model="manualModal" class="modal">
			<template v-slot:header><span class="modal-title">Инструкция по настройке сервера демо-панели</span></template>
			<t-field label="Приватный SSH-ключ"><span class="private" ref="private">a3b3a0f552d65df1eb3fc08b0f8a28854895...</span> <i class="btn-copy"></i> <span class="clickable">Скачать</span></t-field>
			<t-field label="Публичный  SSH-ключ"><span class="public" ref="public">a3b3a0f552d65df1eb3fc08b0f8a28854895...</span> <i class="btn-copy"></i> <span class="clickable">Скачать</span></t-field>
		</at-modal>
	</main>
</template>

<script>
import ServerTable from '@/components/servers/ServerTable.vue'
import NewServer from '@/components/servers/NewServer.vue'

export default {
	name: 'servers',
	components: {
		ServerTable,
		NewServer
	},
	data: () => ({
		addNew: false,
		serverModal: false,
		manualModal: false
	})
}
</script>

<style lang="scss" scoped>
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

.page-servers {
	display: flex;
	height: 100%;
	&__list {
		flex-grow: 1;
		height: 100%;
		background: #17212B;
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

.btn-copy {
  display: inline-block;
  vertical-align: middle;
  cursor: pointer;
  width: 24px;
  height: 24px;
  margin-left: 20px;
  margin-right: 35px;
  background-image: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iI0E3QkVEMyI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xNiAxSDRjLTEuMSAwLTIgLjktMiAydjE0aDJWM2gxMlYxem0zIDRIOGMtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxMWMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0wIDE2SDhWN2gxMXYxNHoiLz48L3N2Zz4=');
}

.ssh {
	text-overflow: ellipsis;
	overflow: hidden;
	max-width: 300px;
}
</style>