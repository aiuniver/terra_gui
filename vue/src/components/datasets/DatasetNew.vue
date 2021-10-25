<template>
	<div class="datasets">
		<p class="datasets__type">{{ list[selectedType] }} <i class="ci-icon ci-file" v-show="selectedType === 1" /></p>
		<div class="datasets__filter">
			<DInputText small icon="search" v-model.trim="search" class="datasets__filter--search" placeholder="Найти в списке"/>
			<div class="datasets__filter--sort">
				<p><span>Последние просмотренные</span> <i class="ci-icon ci-chevron_down" /></p>
				<i class="ci-icon ci-tile" />
				<i class="ci-icon ci-text_align_justify" />
			</div>
		</div>
		<scrollbar class="datasets__scroll">
			<div class="datasets__list">
				<CardDataset v-for="(item, idx) in sortedList" :key="idx" :dataset="item"/>
			</div>
		</scrollbar>
	</div>
</template>

<script>
import CardDataset from './cards/CardDatasetNew.vue'

export default {
	components: {
		CardDataset,
		DInputText: () => import('@/components/global/design/forms/components/DInputText')
	},
	data: () => ({
		list: ['Недавние датасеты', 'Проектные датасеты', 'Датасеты Terra'],
		search: ''
	}),
	props: ['datasets', 'selectedType'],
	computed: {
		sortedList() {
			return this.datasets.filter(item => {
				return item.name.toLowerCase().includes(this.search.toLowerCase())
			})
		}
	}
}
</script>

<style lang="scss" scoped>
.datasets {
	padding: 30px 0 0 40px;
	width: 100%;
	height: 100%;
	display: flex;
	flex-direction: column;
	gap: 30px;
	&__list {
		display: flex;
		gap: 20px;
		flex-wrap: wrap;
		margin-bottom: 20px;
	}
	&__type {
		font-size: 14px;
		font-weight: 600;
	}
	&__filter {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-right: 60px;
		&--search {
			max-width: 426px;
		}
		&--sort {
			display: flex;
			gap: 20px;
			align-items: center;
			p {
				color: #A7BED3;
				font-size: 14px;
				* {
					vertical-align: middle;
				}
			}
			i {
				color: #A7BED3;
				font-size: 20px;
				cursor: pointer;
			}
		}
	}
	&__scroll {
		justify-self: stretch;
	}
}

.ci-tile {
	display: inline-block;
	background-repeat: no-repeat;
	border: 1px solid #65B9F4;
	border-radius: 4px;
	background-position: center;
	background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEzLjI1IDEzLjI1SDguNzVWOC43NUgxMy4yNVYxMy4yNVpNNS4yNSAxMy4yNUgwLjc1VjguNzVINS4yNVYxMy4yNVpNMTMuMjUgNS4yNUg4Ljc1VjAuNzVIMTMuMjVWNS4yNVpNNS4yNSA1LjI1SDAuNzVWMC43NUg1LjI1VjUuMjVaIiBzdHJva2U9IiM2NUI5RjQiIHN0cm9rZS13aWR0aD0iMS41Ii8+Cjwvc3ZnPgo=');
	width: 24px;
	height: 24px;
}

.ci-file {
	display: inline-block;
	width: 20px;
	height: 20px;
	margin-left: 10px;
	vertical-align: middle;
	background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzI2NDQ6NDQzMDYpIj4KPHBhdGggZD0iTTE1LjAwMDIgMTguMzMzM0g1LjAwMDE2QzQuMDc5NjkgMTguMzMzMyAzLjMzMzUgMTcuNTg3MSAzLjMzMzUgMTYuNjY2NlYzLjMzMzI5QzMuMzMzNSAyLjQxMjgyIDQuMDc5NjkgMS42NjY2MyA1LjAwMDE2IDEuNjY2NjNIMTAuODMzNUMxMC44NDI2IDEuNjY3MDIgMTAuODUxNiAxLjY2ODcgMTAuODYwMiAxLjY3MTYzQzEwLjg2ODMgMS42NzQxNCAxMC44NzY3IDEuNjc1ODIgMTAuODg1MiAxLjY3NjYzQzEwLjk1ODcgMS42ODEzNCAxMS4wMzEyIDEuNjk1NjIgMTEuMTAxIDEuNzE5MTNMMTEuMTI0MyAxLjcyNjYzQzExLjE0MzIgMS43MzMwMyAxMS4xNjE2IDEuNzQwODMgMTEuMTc5MyAxLjc0OTk2QzExLjI3MDEgMS43OTAzMSAxMS4zNTI4IDEuODQ2NzggMTEuNDIzNSAxLjkxNjYzTDE2LjQyMzUgNi45MTY2M0MxNi40OTMzIDYuOTg3MjggMTYuNTQ5OCA3LjA3MDAxIDE2LjU5MDIgNy4xNjA3OUMxNi41OTg1IDcuMTc5MTMgMTYuNjA0MyA3LjE5ODI5IDE2LjYxMSA3LjIxNzQ2TDE2LjYxODUgNy4yMzkxM0MxNi42NDE4IDcuMzA4NjIgMTYuNjU1NSA3LjM4MDk0IDE2LjY1OTMgNy40NTQxM0MxNi42NjA2IDcuNDYxNzggMTYuNjYyNiA3LjQ2OTMxIDE2LjY2NTIgNy40NzY2M0MxNi42NjY2IDcuNDg0MzEgMTYuNjY3MiA3LjQ5MjE0IDE2LjY2NjkgNy40OTk5NlYxNi42NjY2QzE2LjY2NjkgMTcuNTg3MSAxNS45MjA2IDE4LjMzMzMgMTUuMDAwMiAxOC4zMzMzWk01LjAwMDE2IDMuMzMzMjlWMTYuNjY2NkgxNS4wMDAyVjguMzMzMjlIMTAuODMzNUMxMC4zNzMzIDguMzMzMjkgMTAuMDAwMiA3Ljk2MDIgMTAuMDAwMiA3LjQ5OTk2VjMuMzMzMjlINS4wMDAxNlpNMTEuNjY2OCA0LjUxMTYzVjYuNjY2NjNIMTMuODIxOEwxMS42NjY4IDQuNTExNjNaIiBmaWxsPSIjQTdCRUQzIi8+CjxwYXRoIGQ9Ik02LjE2NjUgOC42NjY2M0gtMC44MzM0OTZILTEuMzMzNVY5LjE2NjYzVjEwLjgzMzNWMTEuMzMzM0gtMC44MzM0OTZINi4xNjY1VjEzLjMzMzNWMTQuMzczNkw2Ljk3ODg1IDEzLjcyMzdMMTEuMTQ1NSAxMC4zOTA0TDExLjYzMzYgOS45OTk5NkwxMS4xNDU1IDkuNjA5NTJMNi45Nzg4NSA2LjI3NjE5TDYuMTY2NSA1LjYyNjMxVjYuNjY2NjNWOC42NjY2M1oiIGZpbGw9IiNBN0JFRDMiIHN0cm9rZT0iIzE3MjEyQiIvPgo8L2c+CjxkZWZzPgo8Y2xpcFBhdGggaWQ9ImNsaXAwXzI2NDQ6NDQzMDYiPgo8cmVjdCB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIGZpbGw9IndoaXRlIi8+CjwvY2xpcFBhdGg+CjwvZGVmcz4KPC9zdmc+Cg==');
}
</style>